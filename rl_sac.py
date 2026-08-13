#!/usr/bin/env python3
"""SAC E.V.A — Soft Actor-Critic pour le trading M15 sur latents JEPA.

Module autonome d'apprentissage par renforcement compatible avec
l'infrastructure JEPA existante (``jax_arena.py``, ``champion_factory.py``).

Approche
--------
L'agent observe l'état latent JEPA H_t (128 dims) + sa position + une
fenêtre de retours récents, et émet une action CONTINUE a ∈ [-1, 1]
(position cible) via une politique stochastique gaussienne squasheé par
tanh (ré-paramétrisation). La récompense par barre M15 est identique à
``jax_arena._pas_simulation`` (position × rendement − coût de transaction).

Algorithme : SAC (Haarnoja et al. 2018)
- entropie maximale : objectif J(π) = Σ E[r + α·H(π)] (exploration robuste),
- double critique (clipped double Q-learning),
- température α auto-ajustée (target entropy = -dim_action),
- politique stochastique avec ré-paramétrisation + tanh squashing,
- réseaux cibles par copie douce (Polyak τ=0.005),
- replay buffer FIFO.

Sorties (conventions ``champion_factory.py``)
---------------------------------------------
- registry_rl/<run_id>/registry.jsonl       champions validés holdout
- registry_rl/<run_id>/champions/*.npz      poids aplatis p0..pN + métriques
- registry_rl/<run_id>/run_meta.json        config + métriques finales

Usage
-----
    # Test rapide (10 s) — vérifie que tout tourne sans crash :
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_sac.py --test

    # Entraînement complet :
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_sac.py \
        --symbole US30.cash --run-id sac_us30_v1 --steps 200000 --segment 512

    # Via config JSON :
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_sac.py --config configs_rl/sac_US30.json

Hyperparamètres principaux (défauts robustes)
---------------------------------------------
lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2 (ou auto=True),
entropie_cible=-1, buffer_taille=100000, batch=64, largeur=256.

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.rl.sac")

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------
DIM_LATENT: int = 128
CAPITAL_INITIAL: float = 100_000.0
COUT_TRANSACTION: float = 0.0002
LEVIER: float = 1.0
FENETRE_SORTINO: int = 256
EPS: float = 1e-8

DD_MAX_HOLDOUT: float = 5.0
NP_MIN_HOLDOUT: float = 0.0


# ---------------------------------------------------------------------------
# Environnement de trading — version CONTINUE (identique à rl_td3.py)
# ---------------------------------------------------------------------------
class EnvironnementTrading:
    """MDP de trading avec action continue (position cible ∈ [-1, 1]).

    La récompense par barre reproduit exactement la simulation de
    ``jax_arena._pas_simulation`` : rendement de la position courante
    (mark-to-market) moins le coût de transaction du changement de
    position.
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
            capital_initial: Capital de départ.
        """
        self.prix = prix.astype(np.float64)
        self.latents = latents.astype(np.float32)
        self.fenetre_retours = fenetre_retours
        self.cout_transaction = cout_transaction
        self.levier = levier
        self.capital_initial = capital_initial
        self.dim_obs = DIM_LATENT + 1 + fenetre_retours
        self.dim_action = 1
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

    def step(self, action: np.ndarray | float) -> tuple[np.ndarray, float, bool]:
        """Exécute une action continue et avance d'une barre M15.

        Args:
            action: Position cible scalaire ∈ [-1, 1] (ou vecteur 1D).

        Returns:
            Tuple ``(observation_suivante, recompense, terminaison)``.
        """
        cible = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        rendement_prix = self.rendements[self.t]
        retour_net = (
            self.position * rendement_prix * self.levier
            - abs(cible - self.position) * self.cout_transaction * self.levier
        )

        self.pnl_trade += retour_net
        changement_signe = (
            self.position != 0.0
            and cible != 0.0
            and np.sign(cible) != np.sign(self.position)
        )
        retour_neutre = self.position != 0.0 and abs(cible) < 1e-6
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
# Réseaux SAC
# ---------------------------------------------------------------------------
class ReseauPolitique(nn.Module):
    """Politique stochastique : gaussienne squasheé par tanh.

    Émet μ et log_σ puis applique la ré-paramétrisation
    a = tanh(μ + σ·ε) avec correction du log-probabilité.

    Attributes:
        tronc: ``dim_obs -> 256 -> 256``.
        tete_moyenne: ``256 -> 1``.
        tete_log_ecart: ``256 -> 1``.
    """

    def __init__(self, dim_obs: int, dim_action: int = 1, largeur: int = 256,
                 log_ecart_min: float = -20.0, log_ecart_max: float = 2.0) -> None:
        """Initialise la politique.

        Args:
            dim_obs: Dimension d'observation.
            dim_action: Dimension d'action (1).
            largeur: Largeur des couches cachées.
            log_ecart_min: Borne inférieure de log_σ.
            log_ecart_max: Borne supérieure de log_σ.
        """
        super().__init__()
        self.dim_action = dim_action
        self.log_ecart_min = log_ecart_min
        self.log_ecart_max = log_ecart_max
        self.tronc = nn.Sequential(
            nn.Linear(dim_obs, largeur),
            nn.ReLU(),
            nn.Linear(largeur, largeur),
            nn.ReLU(),
        )
        self.tete_moyenne = nn.Linear(largeur, dim_action)
        self.tete_log_ecart = nn.Linear(largeur, dim_action)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Échantillonne une action et calcule son log-prob.

        Args:
            x: ``(batch, dim_obs)``.

        Returns:
            Tuple ``(action_squashee, log_prob, moyenne, log_ecart)``.
        """
        h = self.tronc(x)
        moyenne = self.tete_moyenne(h)
        log_ecart = torch.clamp(
            self.tete_log_ecart(h), self.log_ecart_min, self.log_ecart_max
        )
        ecart = log_ecart.exp()
        eps = torch.randn_like(moyenne)
        action_brute = moyenne + ecart * eps
        action = torch.tanh(action_brute)
        # log π(a) avec correction du jacobien tanh.
        log_prob = (
            -0.5 * ((action_brute - moyenne) / (ecart + 1e-8)) ** 2
            - log_ecart
            - math.log(math.sqrt(2.0 * math.pi))
        ).sum(dim=-1, keepdim=True)
        log_prob = log_prob - torch.log(1.0 - action**2 + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob, moyenne, log_ecart


class Critique(nn.Module):
    """MLP Q(s, a) : ``dim_obs + 1 -> 256 -> 256 -> 1``.

    Attributes:
        couches: Séquentiel.
    """

    def __init__(self, dim_obs: int, largeur: int = 256) -> None:
        """Initialise le critique.

        Args:
            dim_obs: Dimension d'observation.
            largeur: Largeur des couches cachées.
        """
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(dim_obs + 1, largeur),
            nn.ReLU(),
            nn.Linear(largeur, largeur),
            nn.ReLU(),
            nn.Linear(largeur, 1),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Calcule Q(s, a) pour un batch.

        Args:
            x: ``(batch, dim_obs)``.
            a: ``(batch, 1)``.

        Returns:
            Q-valeurs ``(batch, 1)``.
        """
        return self.couches(torch.cat([x, a], dim=-1))


# ---------------------------------------------------------------------------
# Replay buffer (identique à rl_td3.py)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Buffer d'expérience FIFO (s, a, r, s', fini).

    Attributes:
        taille_max: Capacité maximale.
    """

    def __init__(self, taille_max: int = 100_000) -> None:
        """Initialise le buffer.

        Args:
            taille_max: Capacité maximale.
        """
        self.taille_max = taille_max
        self.transitions: deque[tuple] = deque(maxlen=taille_max)

    def ajouter(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        recompense: float,
        obs_suivante: np.ndarray,
        fini: bool,
    ) -> None:
        """Ajoute une transition.

        Args:
            obs: Observation ``(dim_obs,)``.
            action: Action ``(1,)``.
            recompense: Récompense scalaire.
            obs_suivante: Observation suivante.
            fini: Terminaison d'épisode.
        """
        self.transitions.append((obs, action, recompense, obs_suivante, fini))

    def echantillonner(self, batch: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Échantillonne un mini-lot uniforme.

        Args:
            batch: Taille du lot.
            device: Device de sortie.

        Returns:
            Tuple de tenseurs ``(obs, actions, recompenses, obs_suivantes,
            finis)``.
        """
        indices = np.random.choice(len(self.transitions), size=batch, replace=False)
        obs, actions, recompenses, obs_suiv, finis = zip(
            *(self.transitions[i] for i in indices)
        )
        obs_t = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(np.stack(actions), dtype=torch.float32, device=device)
        recompenses_t = torch.as_tensor(recompenses, dtype=torch.float32, device=device)
        obs_suiv_t = torch.as_tensor(np.stack(obs_suiv), dtype=torch.float32, device=device)
        finis_t = torch.as_tensor(finis, dtype=torch.float32, device=device)
        return obs_t, actions_t, recompenses_t, obs_suiv_t, finis_t

    def __len__(self) -> int:
        """Retourne le nombre de transitions stockées."""
        return len(self.transitions)


# ---------------------------------------------------------------------------
# Agent SAC
# ---------------------------------------------------------------------------
class AgentSAC:
    """Agent SAC : entropie maximale, double critique, α auto-ajusté.

    Attributes:
        politique: Réseau de politique stochastique.
        critique1, critique2: Critiques jumeaux.
        alpha: Température (log_alpha si auto).
    """

    def __init__(
        self,
        dim_obs: int,
        dim_action: int = 1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        alpha_auto: bool = True,
        entropie_cible: float | None = None,
        largeur: int = 256,
        device: torch.device | None = None,
    ) -> None:
        """Initialise l'agent.

        Args:
            dim_obs: Dimension d'observation.
            dim_action: Dimension d'action.
            lr: Taux d'apprentissage commun.
            gamma: Facteur d'actualisation.
            tau: Copie douce Polyak.
            alpha: Température initiale.
            alpha_auto: Ajuste α par gradient (SAC auto).
            entropie_cible: Cible d'entropie (défaut = -dim_action).
            largeur: Largeur des MLP.
            device: Device torch.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.politique = ReseauPolitique(dim_obs, dim_action, largeur).to(self.device)
        self.critique1 = Critique(dim_obs, largeur).to(self.device)
        self.critique2 = Critique(dim_obs, largeur).to(self.device)
        self.critique1_cible = Critique(dim_obs, largeur).to(self.device)
        self.critique2_cible = Critique(dim_obs, largeur).to(self.device)
        self.critique1_cible.load_state_dict(self.critique1.state_dict())
        self.critique2_cible.load_state_dict(self.critique2.state_dict())

        self.optim_politique = torch.optim.Adam(self.politique.parameters(), lr=lr)
        self.optim_critiques = torch.optim.Adam(
            list(self.critique1.parameters()) + list(self.critique2.parameters()), lr=lr
        )
        self.gamma = gamma
        self.tau = tau

        # Température.
        self.alpha_auto = alpha_auto
        if alpha_auto:
            self.log_alpha = torch.tensor(math.log(alpha), requires_grad=True,
                                          device=self.device)
            self.optim_alpha = torch.optim.Adam([self.log_alpha], lr=lr)
            self.entropie_cible = (
                -float(dim_action) if entropie_cible is None else float(entropie_cible)
            )
        else:
            self.log_alpha = torch.tensor(math.log(alpha), device=self.device)
            self.optim_alpha = None

        self.moyenne_obs = np.zeros(dim_obs, dtype=np.float32)
        self.var_obs = np.ones(dim_obs, dtype=np.float32)
        self.nb_obs = 0

    @property
    def alpha(self) -> torch.Tensor:
        """Température courante (exponentielle de log_alpha).

        Returns:
            Tenseur scalaire.
        """
        return self.log_alpha.exp()

    def normaliser_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalise une observation par running stats.

        Args:
            obs: Observation ou batch.

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
    ) -> np.ndarray:
        """Choisit une action (stochastique ou moyenne déterministe).

        Args:
            obs: Observation ``(dim_obs,)``.
            deterministe: True = moyenne (tanh(μ)) sans bruit.

        Returns:
            Action ``(1,)`` ∈ [-1, 1].
        """
        obs_n = self.normaliser_obs(obs)
        x = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, log_prob, moyenne, _ = self.politique(x)
            if deterministe:
                return torch.tanh(moyenne).cpu().numpy().astype(np.float32)
            return action.cpu().numpy().astype(np.float32)

    def apprendre(self, buffer: ReplayBuffer, batch: int = 64) -> dict[str, float]:
        """Effectue une étape d'apprentissage SAC complète.

        Args:
            buffer: Replay buffer.
            batch: Taille du mini-lot.

        Returns:
            Dictionnaire des pertes (critique1, critique2, politique,
            alpha, alpha_valeur).
        """
        if len(buffer) < batch:
            return {"perte_critiques": 0.0, "perte_politique": 0.0,
                    "perte_alpha": 0.0, "alpha": float(self.alpha.item())}
        obs, actions, recompenses, obs_suiv, finis = buffer.echantillonner(
            batch, self.device
        )
        self.mettre_a_jour_normalisation(obs.cpu().numpy())
        alpha = self.alpha

        # --- Critiques : soft target = r + γ(min Q'(s',a') − α log π(a'|s')) ---
        with torch.no_grad():
            actions_suiv, log_probs_suiv, _, _ = self.politique(obs_suiv)
            q1_suiv = self.critique1_cible(obs_suiv, actions_suiv)
            q2_suiv = self.critique2_cible(obs_suiv, actions_suiv)
            q_min = torch.min(q1_suiv, q2_suiv) - alpha * log_probs_suiv
            cibles = recompenses[:, None] + self.gamma * q_min * (1.0 - finis[:, None])

        q1 = self.critique1(obs, actions)
        q2 = self.critique2(obs, actions)
        perte_c1 = F.mse_loss(q1, cibles)
        perte_c2 = F.mse_loss(q2, cibles)
        perte_critiques = perte_c1 + perte_c2
        self.optim_critiques.zero_grad()
        perte_critiques.backward()
        nn.utils.clip_grad_norm_(
            list(self.critique1.parameters()) + list(self.critique2.parameters()), 1.0
        )
        self.optim_critiques.step()

        # --- Politique : maximiser E[Q(s,a) − α log π(a|s)] -----------------
        actions_nouv, log_probs_nouv, _, _ = self.politique(obs)
        q1_nouv = self.critique1(obs, actions_nouv)
        q2_nouv = self.critique2(obs, actions_nouv)
        q_min_nouv = torch.min(q1_nouv, q2_nouv)
        perte_politique = (alpha.detach() * log_probs_nouv - q_min_nouv).mean()
        self.optim_politique.zero_grad()
        perte_politique.backward()
        nn.utils.clip_grad_norm_(self.politique.parameters(), 1.0)
        self.optim_politique.step()

        # --- Température : minimiser α·(−log π − H_cible) --------------------
        perte_alpha = torch.as_tensor(0.0, device=self.device)
        if self.alpha_auto:
            perte_alpha = (
                -self.log_alpha * (log_probs_nouv.detach() + self.entropie_cible)
            ).mean()
            self.optim_alpha.zero_grad()
            perte_alpha.backward()
            self.optim_alpha.step()

        # --- Copie douce ------------------------------------------------------
        with torch.no_grad():
            for cible, source in [
                (self.critique1_cible, self.critique1),
                (self.critique2_cible, self.critique2),
            ]:
                for p_cible, p in zip(cible.parameters(), source.parameters()):
                    p_cible.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

        return {
            "perte_critiques": float(perte_critiques.item()),
            "perte_politique": float(perte_politique.item()),
            "perte_alpha": float(perte_alpha.item()),
            "alpha": float(alpha.item()),
        }


# ---------------------------------------------------------------------------
# Sauvegarde / chargement des champions (.npz compatible JEPA)
# ---------------------------------------------------------------------------
def sauvegarder_champion(
    agent: AgentSAC,
    chemin: Path,
    metriques: dict[str, float],
    extra: dict | None = None,
) -> Path:
    """Sauvegarde les poids de la politique en ``.npz`` (flat p0..pN).

    Le champion sauvegardé est la POLITIQUE (mean network pour
    l'exploitation). Les critiques et la température ne sont pas persistés.

    Args:
        agent: Agent dont on sauvegarde la politique.
        chemin: Chemin du fichier de sortie.
        metriques: Métriques embarquées.
        extra: Métadonnées additionnelles.

    Returns:
        Chemin du fichier écrit.
    """
    chemin.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().numpy() for k, v in agent.politique.state_dict().items()}
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


def charger_champion(chemin: str | Path, dim_obs: int = DIM_LATENT + 17) -> ReseauPolitique:
    """Recharge un champion SAC (politique) depuis un ``.npz``.

    Args:
        chemin: Fichier ``.npz`` produit par ``sauvegarder_champion``.
        dim_obs: Dimension d'observation attendue.

    Returns:
        Réseau de politique chargé (évaluation).

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si le nombre de feuilles ne correspond pas.
    """
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"Champion introuvable : {chemin}")
    donnees = np.load(chemin, allow_pickle=True)
    cles_ord = sorted(
        [k for k in donnees.keys() if re.match(r"p\d+$", k)],
        key=lambda k: int(k[1:]),
    )
    politique = ReseauPolitique(dim_obs, 1)
    ordre = list(politique.state_dict().keys())
    if len(cles_ord) != len(ordre):
        raise ValueError(
            f"{len(cles_ord)} feuilles dans le .npz, {len(ordre)} attendues "
            f"(dim_obs={dim_obs})"
        )
    state = {nom: torch.as_tensor(donnees[cle]) for nom, cle in zip(ordre, cles_ord)}
    politique.load_state_dict(state)
    politique.eval()
    return politique


# ---------------------------------------------------------------------------
# Évaluation holdout
# ---------------------------------------------------------------------------
def evaluer_politique(
    agent: AgentSAC,
    prix: np.ndarray,
    latents: np.ndarray,
    t_debut: int | None = None,
    nb_pas: int = 1500,
    nb_episodes: int = 3,
) -> dict[str, float]:
    """Évalue la politique (déterministe, tanh(μ)) sur un segment holdout.

    Args:
        agent: Agent entraîné.
        prix: Prix de la séquence ``(nb,)``.
        latents: Latents ``(nb, 128)``.
        t_debut: Barre de départ (None = aléatoire, moyenne).
        nb_pas: Nombre de pas par épisode.
        nb_episodes: Nombre d'épisodes moyennés.

    Returns:
        Métriques moyennées.
    """
    env = EnvironnementTrading(prix, latents)
    totaux: dict[str, float] = {}
    for _ in range(nb_episodes):
        obs = env.reset(t_debut)
        for _pas in range(nb_pas):
            action = agent.choisir_action(obs, deterministe=True)
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
    """Exécute un run d'entraînement SAC complet.

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
        "SAC %s | %s | train=%d holdout=%d | device=%s",
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

    agent = AgentSAC(
        dim_obs=dim_obs,
        dim_action=1,
        lr=float(cfg.get("lr", 3e-4)),
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.005)),
        alpha=float(cfg.get("alpha", 0.2)),
        alpha_auto=bool(cfg.get("alpha_auto", True)),
        entropie_cible=(
            float(cfg["entropie_cible"]) if "entropie_cible" in cfg else None
        ),
        largeur=int(cfg.get("largeur", 256)),
    )
    buffer = ReplayBuffer(int(cfg.get("buffer_taille", 100_000)))
    batch = int(cfg.get("batch", 64))
    prefill = int(cfg.get("prefill", 1000))

    env = EnvironnementTrading(
        prix_train,
        latents_train,
        fenetre_retours=fenetre_retours,
        cout_transaction=float(cfg.get("cout_transaction", COUT_TRANSACTION)),
        levier=float(cfg.get("levier", LEVIER)),
    )

    steps = int(cfg.get("steps", 200_000))
    eval_holdout = int(cfg.get("eval_holdout", 5000))
    apprendre_apres = int(cfg.get("apprendre_apres", 2))

    nb_champions = 0
    nb_records = 0
    meilleur_holdout = -np.inf
    meilleures_metriques: dict[str, float] = {}

    # --- Pré-remplissage du buffer (actions aléatoires) -------------------
    obs = env.reset()
    for _ in range(prefill):
        action = np.random.uniform(-1.0, 1.0, size=1).astype(np.float32)
        obs_suiv, r, fini = env.step(action)
        buffer.ajouter(obs, action, r, obs_suiv, fini)
        obs = env.reset() if fini else obs_suiv

    for pas in range(1, steps + 1):
        action = agent.choisir_action(obs)
        obs_suiv, r, fini = env.step(action)
        buffer.ajouter(obs, action, r, obs_suiv, fini)

        pertes: dict[str, float] = {}
        if pas % apprendre_apres == 0:
            for _ in range(apprendre_apres):
                pertes = agent.apprendre(buffer, batch)

        if fini:
            obs = env.reset()
        else:
            obs = obs_suiv

        # Validation holdout périodique.
        if pas % eval_holdout == 0:
            m_h = evaluer_politique(
                agent, prix_hold, latents_hold, nb_pas=min(1500, len(prix_hold))
            )
            generalise = (
                m_h["net_profit"] > float(cfg.get("np_min_holdout", NP_MIN_HOLDOUT))
                and m_h["max_drawdown"] <= float(cfg.get("dd_max_holdout", DD_MAX_HOLDOUT))
            )
            entree = {
                "run_id": run_id, "pas": pas,
                "perte_critiques": round(pertes.get("perte_critiques", 0.0), 5),
                "perte_politique": round(pertes.get("perte_politique", 0.0), 5),
                "alpha": round(pertes.get("alpha", 0.0), 5),
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
                    dossier_champions / f"champion_p{pas}.npz",
                    m_h,
                    {"algo": "sac", "symbole": sym, "pas": pas},
                )
                entree["fichier"] = chemin.name
                with registry_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entree, ensure_ascii=False) + "\n")
                journal.info(
                    "  ✓ pas %d : GÉNÉRALISE np=%+.2f%% dd=%.2f%% wr=%.1f%% pf=%.2f -> %s",
                    pas, m_h["net_profit"], m_h["max_drawdown"],
                    m_h["win_rate"], m_h["profit_factor"], chemin.name,
                )
            else:
                journal.info(
                    "  ✗ pas %d : holdout np=%+.2f%% dd=%.2f%% (fitness=%.3f) α=%.3f",
                    pas, m_h["net_profit"], m_h["max_drawdown"], m_h["fitness"],
                    pertes.get("alpha", 0.0),
                )

        if pas % (eval_holdout * 5) == 0:
            journal.info(
                "run %s pas %6d/%d | buffer=%d | champions=%d | %.0f pas/s",
                run_id, pas, steps, len(buffer), nb_champions,
                pas / max(1e-9, time.perf_counter() - t0),
            )

    # --- Champion final ----------------------------------------------------
    if not meilleures_metriques:
        meilleures_metriques = evaluer_politique(
            agent, prix_hold, latents_hold, nb_pas=min(1500, len(prix_hold))
        )
    chemin_final = sauvegarder_champion(
        agent,
        dossier_champions / "champion_final.npz",
        meilleures_metriques,
        {"algo": "sac", "symbole": sym, "pas": steps},
    )
    entree_final = {
        "run_id": run_id, "pas": steps,
        **{k: round(float(v), 4) for k, v in meilleures_metriques.items()},
        "fichier": chemin_final.name,
    }
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entree_final, ensure_ascii=False) + "\n")

    duree = time.perf_counter() - t0
    resume = {
        "run_id": run_id, "symbole": sym, "algo": "sac",
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
    """Test de validation : mini-run SAC sans crash (données synthétiques)."""
    journal.info("=== TEST RAPIDE SAC ===")
    np.random.seed(0)
    torch.manual_seed(0)
    nb = 1200
    prix = np.cumprod(1.0 + np.random.normal(0, 0.001, nb)) * 39000.0
    latents = np.random.normal(0, 1, (nb, DIM_LATENT)).astype(np.float32)
    env = EnvironnementTrading(prix, latents)
    agent = AgentSAC(dim_obs=env.dim_obs, dim_action=1)
    buffer = ReplayBuffer(5000)
    obs = env.reset()
    for pas in range(300):
        action = agent.choisir_action(obs)
        obs_suiv, r, fini = env.step(action)
        buffer.ajouter(obs, action, r, obs_suiv, fini)
        if pas % 2 == 0:
            agent.apprendre(buffer, 32)
        obs = env.reset() if fini else obs_suiv
    m = evaluer_politique(agent, prix, latents, nb_pas=200, nb_episodes=1)
    journal.info("np=%+.2f%% dd=%.2f%% fitness=%.3f α=%.3f",
                 m["net_profit"], m["max_drawdown"], m["fitness"], agent.alpha.item())
    chemin = sauvegarder_champion(
        agent, Path("/tmp/sac_test_champion.npz"),
        {"fitness": 1.0, "net_profit": 0.5, "max_drawdown": 1.0, "sortino": 0.5,
         "win_rate": 50.0, "profit_factor": 1.2, "nb_trades": 10.0},
        {"algo": "sac", "symbole": "TEST"},
    )
    politique = charger_champion(chemin, dim_obs=env.dim_obs)
    with torch.no_grad():
        x = torch.randn(1, env.dim_obs)
        action, log_prob, _, _ = politique(x)
        assert action.shape == (1, 1) and log_prob.shape == (1, 1)
    journal.info("✅ TEST SAC RÉUSSI — champion rechargé, inférence OK (%s)", chemin)


def main() -> None:
    """Point d'entrée CLI."""
    p = argparse.ArgumentParser(description="SAC E.V.A — RL entropie max trading M15 sur latents JEPA")
    p.add_argument("--config", help="Fichier JSON de config (style champion_factory)")
    p.add_argument("--symbole", default="US30.cash", help="Symbole (ex. US30.cash)")
    p.add_argument("--run-id", default=None, help="Identifiant du run")
    p.add_argument("--steps", type=int, default=200_000, help="Pas d'environnement total")
    p.add_argument("--segment", type=int, default=512, help="Longueur des segments (reset)")
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
            "id": args.run_id or f"sac_{args.symbole.replace('.', '_')}",
            "symbole": args.symbole,
            "timeframe": "m15",
            "steps": args.steps,
            "segment": args.segment,
            "seed": args.seed,
        }
    resume = lancer_run(cfg)
    print(json.dumps(resume, ensure_ascii=False))


if __name__ == "__main__":
    main()
