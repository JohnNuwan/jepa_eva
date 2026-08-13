#!/usr/bin/env python3
"""PPO E.V.A — Proximal Policy Optimization pour le trading M15 sur latents JEPA.

Module autonome d'apprentissage par renforcement (RL) compatible avec
l'infrastructure JEPA existante (``jax_arena.py``, ``champion_factory.py``).

Approche
--------
L'agent observe l'état latent JEPA H_t (128 dims) + sa position courante +
une fenêtre de retours récents, choisit une action discrète
{0: SHORT, 1: FLAT, 2: LONG} (position cible ∈ {-1, 0, +1}), et reçoit une
récompense par barre M15 identique à la simulation de ``jax_arena._pas_simulation``
(mark-to-market, coût de transaction proportionnel au changement de position).

Algorithme : PPO (Schulman et al. 2017) avec
- actor-critic partagé (MLP 256-256), sorties softmax + valeur,
- GAE (lambda=0.95) pour l'estimation des avantages,
- clipping du ratio de probabilité (epsilon=0.2),
- entraînement multi-époques par mini-lots sur un buffer de rollout,
- normalisation de l'observation (running mean/std).

Sorties (mêmes conventions que ``champion_factory.py``)
--------------------------------------------------------
- registry_rl/<run_id>/registry.jsonl       champions validés holdout
- registry_rl/<run_id>/champions/*.npz      poids aplatis p0..pN + métriques
- registry_rl/<run_id>/run_meta.json        config + métriques finales

Le champion .npz est compatible avec ``charger_champion()`` de ce module
(flat pytree p0..pN + clés fitness/net_profit/drawdown/sortino/win_rate/
profit_factor/nb_trades), et lisible par ``backtest_validation.py`` de la
famille JEPA (format de clés identique).

Usage
-----
    # Test rapide (10 s) — vérifie que tout tourne sans crash :
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_ppo.py --test

    # Entraînement complet sur un symbole :
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_ppo.py \
        --symbole US30.cash --run-id ppo_us30_v1 --generations 200 \
        --steps 200000 --segment 512

    # Via fichier de config JSON (format identique à champion_factory) :
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_ppo.py --config configs_rl/ppo_US30.json

Hyperparamètres principaux (défauts robustes)
---------------------------------------------
lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, nb_epochs=4,
mini_batch=64, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
nb_envs=8, segment=512, generations=200, eval_holdout=10.

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.rl.ppo")

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------
DIM_LATENT: int = 128
CAPITAL_INITIAL: float = 100_000.0
COUT_TRANSACTION: float = 0.0002
LEVIER: float = 1.0
FENETRE_SORTINO: int = 256
EPS: float = 1e-8

ACTIONS_POSITION: dict[int, float] = {0: -1.0, 1: 0.0, 2: 1.0}
NOM_ACTIONS: list[str] = ["SHORT", "FLAT", "LONG"]

DD_MAX_HOLDOUT: float = 5.0
NP_MIN_HOLDOUT: float = 0.0


# ---------------------------------------------------------------------------
# Environnement de trading (MDP M15, récompense = jax_arena._pas_simulation)
# ---------------------------------------------------------------------------
class EnvironnementTrading:
    """MDP de trading : état latent JEPA + position + retours récents.

    La récompense par barre reproduit exactement la simulation de
    ``jax_arena`` : rendement de la position courante (mark-to-market)
    moins le coût de transaction du changement de position.

    Attributes:
        prix: Prix M15 ``(nb_barres,)``.
        latents: Encodages JEPA ``(nb_barres, DIM_LATENT)``.
        fenetre_retours: Nombre de retours récents dans l'observation.
        dim_obs: Dimension de l'observation.
        positions: Positions cibles possibles (actions).
    """

    def __init__(
        self,
        prix: np.ndarray,
        latents: np.ndarray,
        fenetre_retours: int = 16,
        cout_transaction: float = COUT_TRANSACTION,
        levier: float = LEVIER,
        capital_initial: float = CAPITAL_INITIAL,
    ) -> None:
        """Initialise l'environnement sur un segment de marché.

        Args:
            prix: Prix M15 ``(nb_barres,)``.
            latents: Latents JEPA ``(nb_barres, dim_latent)``.
            fenetre_retours: Fenêtre de retours passés dans l'obs.
            cout_transaction: Coût par unité de changement de position.
            levier: Levier appliqué aux retours.
            capital_initial: Capital de départ (échelle de référence).
        """
        self.prix = prix.astype(np.float64)
        self.latents = latents.astype(np.float32)
        self.fenetre_retours = fenetre_retours
        self.cout_transaction = cout_transaction
        self.levier = levier
        self.capital_initial = capital_initial
        self.dim_obs = DIM_LATENT + 1 + fenetre_retours
        self.nb_barres = len(self.prix)

        rendements = np.zeros_like(self.prix)
        rendements[1:] = np.diff(self.prix) / (self.prix[:-1] + EPS)
        self.rendements = rendements

        self.t = 0
        self.position = 0.0
        self.retours_recents = np.zeros(fenetre_retours, dtype=np.float32)
        self.equity = capital_initial
        self.equity_peak = capital_initial
        self.historique_equity: list[float] = []
        self.pnl_trade = 0.0
        self.nb_trades = 0
        self.trades_gagnants = 0
        self.profit_brut = 0.0
        self.perte_brute = 0.0

    def _construire_obs(self) -> np.ndarray:
        """Construit l'observation ``(dim_obs,)``.

        Returns:
            Vecteur concaténé : latent_t + position + fenêtre de retours.
        """
        latent = self.latents[self.t]
        return np.concatenate(
            [
                latent,
                np.asarray([self.position], dtype=np.float32),
                self.retours_recents,
            ]
        ).astype(np.float32)

    def reset(self, t_debut: int | None = None) -> np.ndarray:
        """Réinitialise l'environnement à une barre donnée.

        Args:
            t_debut: Index de départ (None = barre aléatoire ≥ 32).

        Returns:
            Observation initiale.
        """
        if t_debut is None:
            t_debut = int(np.random.randint(32, max(32, self.nb_barres - 1)))
        self.t = t_debut
        self.position = 0.0
        self.retours_recents = np.zeros(self.fenetre_retours, dtype=np.float32)
        self.equity = self.capital_initial
        self.equity_peak = self.capital_initial
        self.historique_equity = [self.capital_initial]
        self.pnl_trade = 0.0
        self.nb_trades = 0
        self.trades_gagnants = 0
        self.profit_brut = 0.0
        self.perte_brute = 0.0
        return self._construire_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Exécute une action et avance d'une barre M15.

        Récompense = position_courante × rendement_prix × levier
                     − |Δposition| × coût × levier
        (identique à la simulation JAX de l'arène génétique).

        Args:
            action: 0=SHORT, 1=FLAT, 2=LONG.

        Returns:
            Tuple ``(observation_suivante, recompense, terminaison)``.
        """
        cible = ACTIONS_POSITION[int(action)]
        rendement_prix = self.rendements[self.t]
        # Retour du barre écoulée avec la position détenue.
        retour_net = (
            self.position * rendement_prix * self.levier
            - abs(cible - self.position) * self.cout_transaction * self.levier
        )

        # Suivi des trades (fermeture quand la position repasse par zéro
        # ou change de signe), mêmes règles que jax_arena.
        self.pnl_trade += retour_net
        changement_signe = (
            self.position != 0.0
            and cible != 0.0
            and np.sign(cible) != np.sign(self.position)
        )
        retour_neutre = self.position != 0.0 and cible == 0.0
        trade_ferme = changement_signe or retour_neutre
        if trade_ferme:
            self.nb_trades += 1
            if self.pnl_trade > 0:
                self.trades_gagnants += 1
                self.profit_brut += self.pnl_trade
            else:
                self.perte_brute += -self.pnl_trade
            self.pnl_trade = 0.0

        self.position = cible
        self.equity *= 1.0 + retour_net
        self.equity_peak = max(self.equity_peak, self.equity)
        self.historique_equity.append(self.equity)
        self.retours_recents = np.roll(self.retours_recents, -1)
        self.retours_recents[-1] = retour_net
        self.t += 1

        fin = self.t >= self.nb_barres - 1
        return self._construire_obs(), float(retour_net), fin

    def metriques(self) -> dict[str, float]:
        """Calcule les métriques finales (fitness = Sortino×2 − DD + NP).

        Returns:
            Dictionnaire fitness, net_profit (%), max_drawdown (%),
            sortino, win_rate (%), profit_factor, nb_trades.
        """
        equity = np.asarray(self.historique_equity, dtype=np.float64)
        retours = np.diff(equity) / (equity[:-1] + EPS)
        if len(retours) > self.fenetre_retours:
            retours = retours[-self.fenetre_retours:]
        moyenne = retours.mean() if len(retours) else 0.0
        retours_neg = np.minimum(retours, 0.0)
        downside_std = float(np.sqrt(np.mean(retours_neg**2) + 1e-12))
        sortino = moyenne / downside_std if downside_std > 1e-12 else 0.0
        drawdown = (self.equity_peak - self.equity) / (self.equity_peak + EPS)
        max_dd_pct = drawdown * 100.0
        net_profit = (self.equity - self.capital_initial) / self.capital_initial * 100.0
        win_rate = (
            self.trades_gagnants / self.nb_trades * 100.0 if self.nb_trades > 0 else 0.0
        )
        profit_factor = (
            self.profit_brut / (self.perte_brute + 1e-9) if self.perte_brute > 0 else 0.0
        )
        fitness = sortino * 2.0 - max_dd_pct + net_profit
        return {
            "fitness": fitness,
            "net_profit": net_profit,
            "max_drawdown": max_dd_pct,
            "sortino": sortino,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "nb_trades": float(self.nb_trades),
        }


# ---------------------------------------------------------------------------
# Réseaux actor-critic
# ---------------------------------------------------------------------------
class ReseauActeurCritique(nn.Module):
    """MLP partagé avec têtes politique (softmax) et valeur.

    Attributes:
        tronc: Tronc commun ``dim_obs -> 256 -> 256``.
        tete_politique: Tête softmax ``256 -> 3``.
        tete_valeur: Tête scalaire ``256 -> 1``.
    """

    def __init__(self, dim_obs: int, dim_action: int = 3, largeur: int = 256) -> None:
        """Initialise le réseau.

        Args:
            dim_obs: Dimension de l'observation.
            dim_action: Nombre d'actions discrètes (3 par défaut).
            largeur: Largeur des couches cachées.
        """
        super().__init__()
        self.tronc = nn.Sequential(
            nn.Linear(dim_obs, largeur),
            nn.Tanh(),
            nn.Linear(largeur, largeur),
            nn.Tanh(),
        )
        self.tete_politique = nn.Linear(largeur, dim_action)
        self.tete_valeur = nn.Linear(largeur, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        """Propagation avant complète.

        Args:
            x: Batch d'observations ``(batch, dim_obs)``.

        Returns:
            Tuple ``(distribution_categorielle, valeur)``.
        """
        h = self.tronc(x)
        logits = self.tete_politique(h)
        valeur = self.tete_valeur(h).squeeze(-1)
        return torch.distributions.Categorical(logits=logits), valeur


# ---------------------------------------------------------------------------
# Buffer de rollout PPO
# ---------------------------------------------------------------------------
@dataclass
class BufferRollout:
    """Stocke un rollout complet pour les mises à jour PPO.

    Attributes:
        observations: ``(T, dim_obs)``.
        actions: ``(T,)`` entiers.
        log_probs: ``(T,)``.
        retours: ``(T,)`` récompenses brutes.
        valeurs: ``(T,)`` prédictions de valeur.
        avantages: ``(T,)`` remplis après GAE.
        retours_actualises: ``(T,)`` remplis après GAE.
    """

    observations: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    actions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    log_probs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    retours: np.ndarray = field(default_factory=lambda: np.zeros(0))
    valeurs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    avantages: np.ndarray = field(default_factory=lambda: np.zeros(0))
    retours_actualises: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def vider(self) -> None:
        """Remet le buffer à zéro."""
        self.observations = np.zeros((0, 0))
        self.actions = np.zeros(0, dtype=np.int64)
        self.log_probs = np.zeros(0)
        self.retours = np.zeros(0)
        self.valeurs = np.zeros(0)
        self.avantages = np.zeros(0)
        self.retours_actualises = np.zeros(0)


# ---------------------------------------------------------------------------
# Agent PPO
# ---------------------------------------------------------------------------
class AgentPPO:
    """Agent PPO : collecte de rollout, GAE, mises à jour par mini-lots.

    Attributes:
        reseau: Actor-critic.
        optimiseur: Adam.
        gamma: Facteur d'actualisation.
        gae_lambda: Paramètre λ de GAE.
        clip_ratio: Borne de clipping du ratio de probabilité.
        ent_coef: Poids du bonus d'entropie.
        vf_coef: Poids de la perte de valeur.
        nb_epochs: Passes d'entraînement par rollout.
        mini_batch: Taille des mini-lots.
        max_grad_norm: Norme de gradient maximale.
        normalisateur_obs: Running mean/std des observations.
    """

    def __init__(
        self,
        dim_obs: int,
        dim_action: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        nb_epochs: int = 4,
        mini_batch: int = 64,
        max_grad_norm: float = 0.5,
        largeur: int = 256,
        device: torch.device | None = None,
    ) -> None:
        """Initialise l'agent.

        Args:
            dim_obs: Dimension d'observation.
            dim_action: Nombre d'actions.
            lr: Taux d'apprentissage.
            gamma: Facteur d'actualisation.
            gae_lambda: λ de GAE.
            clip_ratio: ε du clipping PPO.
            ent_coef: Poids entropie.
            vf_coef: Poids valeur.
            nb_epochs: Époques par rollout.
            mini_batch: Taille de mini-lot.
            max_grad_norm: Clipping de gradient.
            largeur: Largeur du MLP.
            device: Device torch.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reseau = ReseauActeurCritique(dim_obs, dim_action, largeur).to(self.device)
        self.optimiseur = torch.optim.Adam(self.reseau.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.nb_epochs = nb_epochs
        self.mini_batch = mini_batch
        self.max_grad_norm = max_grad_norm
        self.dim_obs = dim_obs
        self.dim_action = dim_action

        self.moyenne_obs = np.zeros(dim_obs, dtype=np.float32)
        self.var_obs = np.ones(dim_obs, dtype=np.float32)
        self.nb_obs = 0

    def normaliser_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalise une observation (ou batch) par running stats.

        Args:
            obs: Observation ou batch ``(..., dim_obs)``.

        Returns:
            Observation normalisée.
        """
        return (obs - self.moyenne_obs) / (np.sqrt(self.var_obs) + 1e-8)

    def mettre_a_jour_normalisation(self, obs: np.ndarray) -> None:
        """Met à jour les running mean/var (Welford).

        Args:
            obs: Batch d'observations ``(T, dim_obs)``.
        """
        batch = obs.shape[0]
        moyenne_batch = obs.mean(axis=0)
        var_batch = obs.var(axis=0)
        delta = moyenne_batch - self.moyenne_obs
        total = self.nb_obs + batch
        self.moyenne_obs += delta * batch / total
        m_a = self.var_obs * self.nb_obs
        m_b = var_batch * batch
        m_2 = m_a + m_b + delta**2 * self.nb_obs * batch / total
        self.var_obs = m_2 / total
        self.nb_obs = total

    @torch.no_grad()
    def choisir_action(
        self, obs: np.ndarray, deterministe: bool = False
    ) -> tuple[int, float]:
        """Choisit une action depuis une observation unique.

        Args:
            obs: Observation ``(dim_obs,)``.
            deterministe: True = argmax (exploitation pure).

        Returns:
            Tuple ``(action, log_prob)``.
        """
        obs_n = self.normaliser_obs(obs)
        x = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        dist, _ = self.reseau(x)
        if deterministe:
            action = int(dist.logits.argmax().item())
            return action, 0.0
        action_t = dist.sample()
        return int(action_t.item()), float(dist.log_prob(action_t).item())

    def collecter_rollout(
        self,
        env: EnvironnementTrading,
        nb_envs: int,
        pas_par_env: int,
        t_debuts: np.ndarray | None = None,
    ) -> BufferRollout:
        """Collecte ``nb_envs × pas_par_env`` transitions en parallèle.

        Les environnements partagent le même tableau de prix/latents mais
        démarrent à des barres distinctes (échantillonnage de segments).

        Args:
            env: Environnement (servira de gabarit).
            nb_envs: Nombre d'environnements parallèles.
            pas_par_env: Longueur de chaque segment.
            t_debuts: Barres de départ (None = aléatoires).

        Returns:
            Buffer de rollout rempli.
        """
        # États parallèles : prix/latents partagés, position propre.
        prix = env.prix
        latents = env.latents
        rendements = env.rendements
        fenetre = env.fenetre_retours

        if t_debuts is None:
            t_debuts = np.random.randint(
                32, max(32, env.nb_barres - pas_par_env - 1), size=nb_envs
            )
        t = t_debuts.copy()
        positions = np.zeros(nb_envs, dtype=np.float64)
        retours_recents = np.zeros((nb_envs, fenetre), dtype=np.float32)
        pnl_trades = np.zeros(nb_envs)
        nb_trades = np.zeros(nb_envs, dtype=np.int64)
        gagnants = np.zeros(nb_envs, dtype=np.int64)
        profit_brut = np.zeros(nb_envs)
        perte_brute = np.zeros(nb_envs)
        equities = np.full(nb_envs, env.capital_initial, dtype=np.float64)
        equity_peaks = np.full(nb_envs, env.capital_initial, dtype=np.float64)

        buffer = BufferRollout()
        obs_buffer = np.zeros((nb_envs * pas_par_env, env.dim_obs), dtype=np.float32)
        action_buffer = np.zeros(nb_envs * pas_par_env, dtype=np.int64)
        logp_buffer = np.zeros(nb_envs * pas_par_env, dtype=np.float32)
        retour_buffer = np.zeros(nb_envs * pas_par_env, dtype=np.float32)
        valeur_buffer = np.zeros(nb_envs * pas_par_env, dtype=np.float32)
        etats_buffer = np.zeros((nb_envs * pas_par_env, env.dim_obs), dtype=np.float32)

        idx = 0
        for pas in range(pas_par_env):
            # Observation courante de chaque env : latent + position + retours.
            latents_t = latents[t]  # (nb_envs, 128)
            obs = np.concatenate(
                [
                    latents_t,
                    positions.astype(np.float32)[:, None],
                    retours_recents,
                ],
                axis=1,
            )
            obs_n = self.normaliser_obs(obs)
            x = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                dist, valeur = self.reseau(x)
                actions_t = dist.sample()
                logps = dist.log_prob(actions_t)
            actions = actions_t.cpu().numpy().astype(np.int64)
            valeurs = valeur.cpu().numpy().astype(np.float32)

            # Stockage.
            tranche = slice(idx, idx + nb_envs)
            obs_buffer[tranche] = obs
            action_buffer[tranche] = actions
            logp_buffer[tranche] = logps.cpu().numpy().astype(np.float32)
            valeur_buffer[tranche] = valeurs
            etats_buffer[tranche] = obs_n

            # Simulation (vectorisée numpy, mêmes règles que l'env scalaire).
            cibles = np.asarray([ACTIONS_POSITION[a] for a in actions])
            rendements_prix = rendements[t]
            retour_net = (
                positions * rendements_prix * env.levier
                - np.abs(cibles - positions) * env.cout_transaction * env.levier
            )
            retour_buffer[tranche] = retour_net.astype(np.float32)

            # Suivi trades.
            pnl_trades += retour_net
            changement_signe = (positions != 0.0) & (cibles != 0.0) & (
                np.sign(cibles) != np.sign(positions)
            )
            retour_neutre = (positions != 0.0) & (cibles == 0.0)
            trade_ferme = changement_signe | retour_neutre
            nb_trades += trade_ferme.astype(np.int64)
            gagne = trade_ferme & (pnl_trades > 0.0)
            perd = trade_ferme & (pnl_trades <= 0.0)
            gagnants += gagne.astype(np.int64)
            profit_brut += np.where(gagne, pnl_trades, 0.0)
            perte_brute += np.where(perd, -pnl_trades, 0.0)
            pnl_trades = np.where(trade_ferme, 0.0, pnl_trades)

            positions = cibles
            equities *= 1.0 + retour_net
            equity_peaks = np.maximum(equity_peaks, equities)
            retours_recents = np.roll(retours_recents, -1, axis=1)
            retours_recents[:, -1] = retour_net
            t = t + 1
            idx += nb_envs

        buffer.observations = obs_buffer
        buffer.actions = action_buffer
        buffer.log_probs = logp_buffer
        buffer.retours = retour_buffer
        buffer.valeurs = valeur_buffer

        # GAE.
        self._calculer_avantages(buffer, valeurs[-nb_envs:] if len(valeurs) else np.zeros(nb_envs))

        # Stats finales pour métriques (env 0 = référence).
        env.t = int(t[0])
        env.position = float(positions[0])
        env.equity = float(equities[0])
        env.equity_peak = float(equity_peaks[0])
        env.retours_recents = retours_recents[0].copy()
        env.historique_equity = list(equities)
        env.nb_trades = int(nb_trades[0])
        env.trades_gagnants = int(gagnants[0])
        env.profit_brut = float(profit_brut[0])
        env.perte_brute = float(perte_brute[0])
        return buffer

    def _calculer_avantages(self, buffer: BufferRollout, valeurs_suivantes: np.ndarray) -> None:
        """Calcule les avantages GAE et les retours actualisés.

        Args:
            buffer: Buffer rempli (retours, valeurs).
            valeurs_suivantes: Valeurs de bootstrap ``(nb_envs,)``.
        """
        T = len(buffer.retours)
        nb_envs = len(valeurs_suivantes)
        avantages = np.zeros(T, dtype=np.float32)
        retours_actualises = np.zeros(T, dtype=np.float32)
        # GAE par environnement (le rollout est concaténé env par env).
        for e in range(nb_envs):
            debut = e * (T // nb_envs)
            fin = (e + 1) * (T // nb_envs)
            gae = 0.0
            for t in range(fin - 1, debut - 1, -1):
                delta = (
                    buffer.retours[t]
                    + self.gamma * buffer.valeurs[min(t + 1, fin - 1)]
                    - buffer.valeurs[t]
                )
                gae = delta + self.gamma * self.gae_lambda * gae
                avantages[t] = gae
                retours_actualises[t] = gae + buffer.valeurs[t]
        buffer.avantages = avantages
        buffer.retours_actualises = retours_actualises

    def mettre_a_jour(self, buffer: BufferRollout) -> dict[str, float]:
        """Effectue les mises à jour PPO (multi-époques, mini-lots).

        Args:
            buffer: Buffer de rollout.

        Returns:
            Dictionnaire des pertes moyennes (politique, valeur, entropie).
        """
        T = len(buffer.observations)
        self.mettre_a_jour_normalisation(buffer.observations)
        obs = torch.as_tensor(
            self.normaliser_obs(buffer.observations), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(buffer.actions, dtype=torch.int64, device=self.device)
        anciens_logps = torch.as_tensor(buffer.log_probs, dtype=torch.float32, device=self.device)
        avantages = torch.as_tensor(buffer.avantages, dtype=torch.float32, device=self.device)
        retours = torch.as_tensor(buffer.retours_actualises, dtype=torch.float32, device=self.device)
        avantages = (avantages - avantages.mean()) / (avantages.std() + 1e-8)

        perte_pol = perte_val = entropie_moy = 0.0
        nb_iter = 0
        for _ in range(self.nb_epochs):
            indices = torch.randperm(T, device=self.device)
            for debut in range(0, T, self.mini_batch):
                lot = indices[debut : debut + self.mini_batch]
                dist, valeur = self.reseau(obs[lot])
                nouveaux_logps = dist.log_prob(actions[lot])
                ratio = torch.exp(nouveaux_logps - anciens_logps[lot])
                perte_adv = -avantages[lot]
                obj_clipe = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                perte_politique = torch.max(ratio * perte_adv, obj_clipe * perte_adv).mean()
                perte_valeur = ((valeur - retours[lot]) ** 2).mean()
                entropie = dist.entropy().mean()
                perte = perte_politique + self.vf_coef * perte_valeur - self.ent_coef * entropie
                self.optimiseur.zero_grad()
                perte.backward()
                nn.utils.clip_grad_norm_(self.reseau.parameters(), self.max_grad_norm)
                self.optimiseur.step()
                perte_pol += float(perte_politique)
                perte_val += float(perte_valeur)
                entropie_moy += float(entropie)
                nb_iter += 1
        return {
            "perte_politique": perte_pol / max(nb_iter, 1),
            "perte_valeur": perte_val / max(nb_iter, 1),
            "entropie": entropie_moy / max(nb_iter, 1),
        }


# ---------------------------------------------------------------------------
# Sauvegarde / chargement des champions (.npz compatible JEPA)
# ---------------------------------------------------------------------------
def sauvegarder_champion(
    agent: AgentPPO,
    chemin: Path,
    metriques: dict[str, float],
    extra: dict | None = None,
) -> Path:
    """Sauvegarde les poids du réseau en ``.npz`` (format flat p0..pN).

    Le format est identique à ``champion_factory.sauvegarder_poids`` :
    clés ``p0..pN`` pour les feuilles aplaties du state_dict + clés
    métriques (fitness, net_profit, drawdown, sortino, win_rate,
    profit_factor, nb_trades).

    Args:
        agent: Agent dont on sauvegarde le réseau.
        chemin: Chemin du fichier de sortie.
        metriques: Métriques embarquées dans le .npz.
        extra: Métadonnées additionnelles (algo, symbole, ...).

    Returns:
        Chemin du fichier écrit.
    """
    chemin.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().numpy() for k, v in agent.reseau.state_dict().items()}
    feuilles = list(state.values())
    donnees = {f"p{i}": feuilles[i] for i in range(len(feuilles))}
    for k, v in metriques.items():
        donnees[k] = np.asarray(float(v))
    if extra:
        for k, v in extra.items():
            if isinstance(v, (int, float, str, bool)):
                donnees[k] = np.asarray(v)
    np.savez_compressed(chemin, **donnees)
    return chemin


def charger_champion(
    chemin: str | Path, dim_obs: int = DIM_LATENT + 17, dim_action: int = 3
) -> ReseauActeurCritique:
    """Recharge un champion PPO depuis un ``.npz``.

    Args:
        chemin: Fichier ``.npz`` produit par ``sauvegarder_champion``.
        dim_obs: Dimension d'observation attendue.
        dim_action: Nombre d'actions.

    Returns:
        Réseau actor-critic chargé (évaluation).

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
    """
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"Champion introuvable : {chemin}")
    donnees = np.load(chemin, allow_pickle=True)
    cles_ord = sorted(
        [k for k in donnees.keys() if re.match(r"p\d+$", k)],
        key=lambda k: int(k[1:]),
    )
    reseau = ReseauActeurCritique(dim_obs, dim_action)
    # Ordre déterministe des clés du state_dict (ordre de déclaration des
    # couches) : tronc.0 / tronc.2 / tete_politique / tete_valeur.
    ordre = list(reseau.state_dict().keys())
    if len(cles_ord) != len(ordre):
        raise ValueError(
            f"{len(cles_ord)} feuilles dans le .npz, {len(ordre)} attendues "
            f"(dim_obs={dim_obs}, dim_action={dim_action})"
        )
    state = {
        nom: torch.as_tensor(donnees[cle]) for nom, cle in zip(ordre, cles_ord)
    }
    reseau.load_state_dict(state)
    reseau.eval()
    return reseau


# ---------------------------------------------------------------------------
# Évaluation holdout
# ---------------------------------------------------------------------------
def evaluer_politique(
    agent: AgentPPO,
    prix: np.ndarray,
    latents: np.ndarray,
    t_debut: int | None = None,
    nb_pas: int = 1500,
    nb_episodes: int = 3,
) -> dict[str, float]:
    """Évalue la politique (déterministe) sur un segment holdout.

    Args:
        agent: Agent entraîné.
        prix: Prix de la séquence ``(nb,)``.
        latents: Latents ``(nb, 128)``.
        t_debut: Barre de départ (None = aléatoire, moyenne sur episodes).
        nb_pas: Nombre de pas par épisode.
        nb_episodes: Nombre d'épisodes moyennés.

    Returns:
        Métriques moyennées (fitness, net_profit, drawdown, sortino,
        win_rate, profit_factor, nb_trades).
    """
    env = EnvironnementTrading(prix, latents)
    totaux: dict[str, float] = {}
    for _ in range(nb_episodes):
        obs = env.reset(t_debut)
        for _pas in range(nb_pas):
            action, _ = agent.choisir_action(obs, deterministe=True)
            obs, _r, fin = env.step(action)
            if fin:
                break
        m = env.metriques()
        for k, v in m.items():
            totaux[k] = totaux.get(k, 0.0) + v / nb_episodes
    return totaux


# ---------------------------------------------------------------------------
# Run complet
# ---------------------------------------------------------------------------
def lancer_run(cfg: dict) -> dict:
    """Exécute un run d'entraînement PPO complet.

    Args:
        cfg: Config (symbole, run_id, hyperparamètres, ...).

    Returns:
        Résumé du run (durée, champions, meilleures métriques).
    """
    t0 = time.perf_counter()
    run_id = cfg["id"]
    sym = cfg["symbole"]
    tf = cfg.get("timeframe", "m15")
    base = Path(cfg.get("base_dir", "registry_rl"))
    dossier_run = base / run_id
    dossier_champions = dossier_run / "champions"
    dossier_run.mkdir(parents=True, exist_ok=True)
    registry_path = dossier_run / "registry.jsonl"
    candidates_path = dossier_run / "candidates.jsonl"

    # --- Données ----------------------------------------------------------
    chemin_latents = Path("latents") / f"{sym}_{tf}_latents.npz"
    if not chemin_latents.is_file():
        raise FileNotFoundError(f"Latents absents : {chemin_latents}")
    donnees = np.load(chemin_latents)
    prix = donnees["prix"].astype(np.float64)
    latents = donnees["latents"].astype(np.float32)
    nb = int(prix.shape[0])

    frac_train = float(cfg.get("frac_train", 0.8))
    limite_train = int(nb * frac_train)
    prix_train, prix_hold = prix[:limite_train], prix[limite_train:]
    latents_train, latents_hold = latents[:limite_train], latents[limite_train:]
    max_holdout = int(cfg.get("max_holdout", 1500))
    if len(prix_hold) > max_holdout:
        prix_hold, latents_hold = prix_hold[-max_holdout:], latents_hold[-max_holdout:]
    journal.info(
        "PPO %s | %s | train=%d holdout=%d | device=%s",
        run_id, sym, limite_train, len(prix_hold),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    )

    # --- Hyperparamètres --------------------------------------------------
    fenetre_retours = int(cfg.get("fenetre_retours", 16))
    dim_obs = DIM_LATENT + 1 + fenetre_retours
    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    agent = AgentPPO(
        dim_obs=dim_obs,
        dim_action=3,
        lr=float(cfg.get("lr", 3e-4)),
        gamma=float(cfg.get("gamma", 0.99)),
        gae_lambda=float(cfg.get("gae_lambda", 0.95)),
        clip_ratio=float(cfg.get("clip_ratio", 0.2)),
        ent_coef=float(cfg.get("ent_coef", 0.01)),
        vf_coef=float(cfg.get("vf_coef", 0.5)),
        nb_epochs=int(cfg.get("nb_epochs", 4)),
        mini_batch=int(cfg.get("mini_batch", 64)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.5)),
        largeur=int(cfg.get("largeur", 256)),
    )

    env = EnvironnementTrading(
        prix_train,
        latents_train,
        fenetre_retours=fenetre_retours,
        cout_transaction=float(cfg.get("cout_transaction", COUT_TRANSACTION)),
        levier=float(cfg.get("levier", LEVIER)),
    )

    generations = int(cfg.get("generations", 200))
    steps = int(cfg.get("steps", 200_000))
    nb_envs = int(cfg.get("nb_envs", 8))
    segment = int(cfg.get("segment", 512))
    eval_holdout = int(cfg.get("eval_holdout", 10))
    pas_par_update = nb_envs * segment
    nb_updates = max(1, steps // pas_par_update)

    nb_champions = 0
    nb_records = 0
    meilleur_holdout = -np.inf
    meilleures_metriques: dict[str, float] = {}

    for update in range(nb_updates):
        buffer = agent.collecter_rollout(env, nb_envs, segment)
        pertes = agent.mettre_a_jour(buffer)
        fitness_train = float(pertes["perte_politique"])

        # Validation holdout périodique.
        if (update + 1) % eval_holdout == 0:
            m_h = evaluer_politique(agent, prix_hold, latents_hold, nb_pas=min(1500, len(prix_hold)))
            generalise = (
                m_h["net_profit"] > float(cfg.get("np_min_holdout", NP_MIN_HOLDOUT))
                and m_h["max_drawdown"] <= float(cfg.get("dd_max_holdout", DD_MAX_HOLDOUT))
            )
            entree = {
                "run_id": run_id, "generation": update,
                "fitness_train": round(fitness_train, 4),
                **{k: round(float(v), 4) for k, v in m_h.items()},
            }
            with candidates_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entree, ensure_ascii=False) + "\n")
            nb_records += 1
            if generalise and m_h["fitness"] > meilleur_holdout:
                meilleur_holdout = m_h["fitness"]
                meilleures_metriques = m_h
                nb_champions += 1
                chemin = sauvegarder_champion(
                    agent,
                    dossier_champions / f"champion_up{update}.npz",
                    m_h,
                    {"algo": "ppo", "symbole": sym, "generation": update},
                )
                entree["fichier"] = chemin.name
                with registry_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entree, ensure_ascii=False) + "\n")
                journal.info(
                    "  ✓ up %d : GÉNÉRALISE np=%+.2f%% dd=%.2f%% wr=%.1f%% pf=%.2f -> %s",
                    update, m_h["net_profit"], m_h["max_drawdown"],
                    m_h["win_rate"], m_h["profit_factor"], chemin.name,
                )
            else:
                journal.info(
                    "  ✗ up %d : holdout np=%+.2f%% dd=%.2f%% (fitness=%.3f)",
                    update, m_h["net_profit"], m_h["max_drawdown"], m_h["fitness"],
                )

        if (update + 1) % 25 == 0:
            journal.info(
                "run %s up %3d/%d | perte_pol=%.3f | champions=%d | %.1f up/s",
                run_id, update + 1, nb_updates, fitness_train, nb_champions,
                (update + 1) / max(1e-9, time.perf_counter() - t0),
            )

    # --- Champion final ----------------------------------------------------
    if not meilleures_metriques:
        meilleures_metriques = evaluer_politique(agent, prix_hold, latents_hold, nb_pas=min(1500, len(prix_hold)))
    chemin_final = sauvegarder_champion(
        agent,
        dossier_champions / "champion_final.npz",
        meilleures_metriques,
        {"algo": "ppo", "symbole": sym, "generation": nb_updates},
    )
    entree_final = {
        "run_id": run_id, "generation": nb_updates,
        **{k: round(float(v), 4) for k, v in meilleures_metriques.items()},
        "fichier": chemin_final.name,
    }
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entree_final, ensure_ascii=False) + "\n")

    duree = time.perf_counter() - t0
    resume = {
        "run_id": run_id, "symbole": sym, "algo": "ppo",
        "duree_s": round(duree, 1), "nb_champions": nb_champions,
        "nb_records": nb_records, "meilleur_holdout": round(float(meilleur_holdout), 4),
        "config": cfg,
    }
    with (dossier_run / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(resume, f, ensure_ascii=False, indent=2)
    journal.info(
        "RUN %s TERMINÉ : %d champions | record_holdout=%.3f | %.0fs",
        run_id, nb_champions, meilleur_holdout, duree,
    )
    return resume


def test_rapide() -> None:
    """Test de validation : mini-run PPO sans crash (données synthétiques)."""
    journal.info("=== TEST RAPIDE PPO ===")
    np.random.seed(0)
    torch.manual_seed(0)
    nb = 1200
    prix = np.cumprod(1.0 + np.random.normal(0, 0.001, nb)) * 39000.0
    latents = np.random.normal(0, 1, (nb, DIM_LATENT)).astype(np.float32)
    env = EnvironnementTrading(prix, latents)
    agent = AgentPPO(dim_obs=env.dim_obs, dim_action=3)
    for i in range(3):
        buffer = agent.collecter_rollout(env, nb_envs=4, pas_par_env=128)
        pertes = agent.mettre_a_jour(buffer)
        m = evaluer_politique(agent, prix, latents, nb_pas=200, nb_episodes=1)
        journal.info("up %d | perte_pol=%.3f | np=%+.2f%% | dd=%.2f%%", i, pertes["perte_politique"], m["net_profit"], m["max_drawdown"])
    chemin = sauvegarder_champion(
        agent, Path("/tmp/ppo_test_champion.npz"),
        {"fitness": 1.0, "net_profit": 0.5, "max_drawdown": 1.0, "sortino": 0.5,
         "win_rate": 50.0, "profit_factor": 1.2, "nb_trades": 10.0},
        {"algo": "ppo", "symbole": "TEST"},
    )
    reseau = charger_champion(chemin, dim_obs=env.dim_obs, dim_action=3)
    with torch.no_grad():
        x = torch.randn(1, env.dim_obs)
        dist, valeur = reseau(x)
        assert dist.probs.shape == (1, 3) and valeur.shape == (1,)
    journal.info("✅ TEST PPO RÉUSSI — champion rechargé, inférence OK (%s)", chemin)


def main() -> None:
    """Point d'entrée CLI."""
    p = argparse.ArgumentParser(description="PPO E.V.A — RL trading M15 sur latents JEPA")
    p.add_argument("--config", help="Fichier JSON de config (style champion_factory)")
    p.add_argument("--symbole", default="US30.cash", help="Symbole (ex. US30.cash)")
    p.add_argument("--run-id", default=None, help="Identifiant du run")
    p.add_argument("--generations", type=int, default=200, help="Nombre d'updates")
    p.add_argument("--steps", type=int, default=200_000, help="Pas d'environnement total")
    p.add_argument("--segment", type=int, default=512, help="Longueur des segments")
    p.add_argument("--nb-envs", type=int, default=8, help="Envs parallèles")
    p.add_argument("--seed", type=int, default=0, help="Seed aléatoire")
    p.add_argument("--test", action="store_true", help="Test rapide de validation")
    args = p.parse_args()

    if args.test:
        test_rapide()
        return

    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    else:
        cfg = {
            "id": args.run_id or f"ppo_{args.symbole.replace('.', '_')}",
            "symbole": args.symbole,
            "timeframe": "m15",
            "generations": args.generations,
            "steps": args.steps,
            "segment": args.segment,
            "nb_envs": args.nb_envs,
            "seed": args.seed,
        }
    resume = lancer_run(cfg)
    print(json.dumps(resume, ensure_ascii=False))


if __name__ == "__main__":
    main()
