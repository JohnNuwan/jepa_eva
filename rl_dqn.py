#!/usr/bin/env python3
"""DQN E.V.A — Deep Q-Network pour le trading M15 sur latents JEPA.

Module autonome d'apprentissage par renforcement compatible avec
l'infrastructure JEPA existante (``jax_arena.py``, ``champion_factory.py``).

Approche
--------
L'agent observe l'état latent JEPA H_t (128 dims) + sa position + une
fenêtre de retours récents, et apprend la fonction Q(s, a) pour 3 actions
discrètes {0: SHORT, 1: FLAT, 2: LONG}. La récompense par barre M15 est
identique à ``jax_arena._pas_simulation`` (mark-to-market + coût de
transaction).

Algorithme : DQN (Mnih et al. 2015) avec
- replay buffer prioritaire simple (échantillonnage uniforme, PER optionnel),
- réseau cible (target network) mis à jour par copie douce (τ) ou dure,
- ε-greedy avec décroissance linéaire de l'exploration,
- Double DQN (van Hasselt et al. 2016) pour réduire le biais de
  surestimation,
- Huber loss (L1 lissée).

Sorties (conventions ``champion_factory.py``)
---------------------------------------------
- registry_rl/<run_id>/registry.jsonl       champions validés holdout
- registry_rl/<run_id>/champions/*.npz      poids aplatis p0..pN + métriques
- registry_rl/<run_id>/run_meta.json        config + métriques finales

Usage
-----
    # Test rapide (10 s) — vérifie que tout tourne sans crash :
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_dqn.py --test

    # Entraînement complet :
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_dqn.py \
        --symbole US30.cash --run-id dqn_us30_v1 --steps 200000 --segment 512

    # Via config JSON :
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_dqn.py --config configs_rl/dqn_US30.json

Hyperparamètres principaux (défauts robustes)
---------------------------------------------
lr=1e-3, gamma=0.99, buffer_taille=100000, batch=64,
tau=0.005 (copie douce), eps_debut=1.0, eps_fin=0.02, eps_decay=4000 pas,
double_dqn=True, update_cible=1000, largeur=256.

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.rl.dqn")

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
# Environnement de trading (identique à rl_ppo.py)
# ---------------------------------------------------------------------------
class EnvironnementTrading:
    """MDP de trading : état latent JEPA + position + retours récents.

    La récompense par barre reproduit exactement la simulation de
    ``jax_arena`` : rendement de la position courante (mark-to-market)
    moins le coût de transaction du changement de position.
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
# Réseau Q
# ---------------------------------------------------------------------------
class ReseauQ(nn.Module):
    """MLP de Q-valeurs : ``dim_obs -> 256 -> 256 -> dim_action``.

    Attributes:
        couches: Séquentiel de couches linéaires + ReLU.
    """

    def __init__(self, dim_obs: int, dim_action: int = 3, largeur: int = 256) -> None:
        """Initialise le réseau.

        Args:
            dim_obs: Dimension d'observation.
            dim_action: Nombre d'actions.
            largeur: Largeur des couches cachées.
        """
        super().__init__()
        self.couches = nn.Sequential(
            nn.Linear(dim_obs, largeur),
            nn.ReLU(),
            nn.Linear(largeur, largeur),
            nn.ReLU(),
            nn.Linear(largeur, dim_action),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calcule les Q-valeurs d'un batch d'observations.

        Args:
            x: ``(batch, dim_obs)``.

        Returns:
            Q-valeurs ``(batch, dim_action)``.
        """
        return self.couches(x)


# ---------------------------------------------------------------------------
# Replay buffer
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
        action: int,
        recompense: float,
        obs_suivante: np.ndarray,
        fini: bool,
    ) -> None:
        """Ajoute une transition.

        Args:
            obs: Observation ``(dim_obs,)``.
            action: Action entière.
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
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=device)
        recompenses_t = torch.as_tensor(recompenses, dtype=torch.float32, device=device)
        obs_suiv_t = torch.as_tensor(np.stack(obs_suiv), dtype=torch.float32, device=device)
        finis_t = torch.as_tensor(finis, dtype=torch.float32, device=device)
        return obs_t, actions_t, recompenses_t, obs_suiv_t, finis_t

    def __len__(self) -> int:
        """Retourne le nombre de transitions stockées."""
        return len(self.transitions)


# ---------------------------------------------------------------------------
# Agent DQN
# ---------------------------------------------------------------------------
class AgentDQN:
    """Agent DQN : ε-greedy, double DQN, réseau cible.

    Attributes:
        reseau: Réseau Q entraîné.
        reseau_cible: Réseau cible (copie).
        optimiseur: Adam.
        gamma: Facteur d'actualisation.
        tau: Coefficient de copie douce.
        eps: Epsilon courant.
        eps_fin: Epsilon minimal.
        eps_decay: Pas d'exploration avant décroissance complète.
        double_dqn: Active le Double DQN.
        normalisateur_obs: Running mean/std.
    """

    def __init__(
        self,
        dim_obs: int,
        dim_action: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        eps_debut: float = 1.0,
        eps_fin: float = 0.02,
        eps_decay: int = 4000,
        double_dqn: bool = True,
        largeur: int = 256,
        device: torch.device | None = None,
    ) -> None:
        """Initialise l'agent.

        Args:
            dim_obs: Dimension d'observation.
            dim_action: Nombre d'actions.
            lr: Taux d'apprentissage.
            gamma: Facteur d'actualisation.
            tau: Copie douce du réseau cible.
            eps_debut: Epsilon initial.
            eps_fin: Epsilon minimal.
            eps_decay: Pas avant décroissance complète.
            double_dqn: Utilise la variante Double DQN.
            largeur: Largeur du MLP.
            device: Device torch.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.reseau = ReseauQ(dim_obs, dim_action, largeur).to(self.device)
        self.reseau_cible = ReseauQ(dim_obs, dim_action, largeur).to(self.device)
        self.reseau_cible.load_state_dict(self.reseau.state_dict())
        self.optimiseur = torch.optim.Adam(self.reseau.parameters(), lr=lr)
        self.perte = nn.SmoothL1Loss()
        self.gamma = gamma
        self.tau = tau
        self.eps = eps_debut
        self.eps_fin = eps_fin
        self.eps_decay = eps_decay
        self.double_dqn = double_dqn

        self.moyenne_obs = np.zeros(dim_obs, dtype=np.float32)
        self.var_obs = np.ones(dim_obs, dtype=np.float32)
        self.nb_obs = 0

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
    ) -> int:
        """Choisit une action ε-greedy.

        Args:
            obs: Observation ``(dim_obs,)``.
            deterministe: True = pas d'exploration (argmax).

        Returns:
            Action entière.
        """
        if not deterministe and np.random.rand() < self.eps:
            return int(np.random.randint(0, self.dim_action))
        obs_n = self.normaliser_obs(obs)
        x = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.reseau(x)
        return int(q.argmax().item())

    def apprendre(self, buffer: ReplayBuffer, batch: int = 64) -> float:
        """Effectue une étape d'apprentissage sur un mini-lot.

        Args:
            buffer: Replay buffer.
            batch: Taille du mini-lot.

        Returns:
            Perte moyenne (0.0 si buffer insuffisant).
        """
        if len(buffer) < batch:
            return 0.0
        obs, actions, recompenses, obs_suiv, finis = buffer.echantillonner(
            batch, self.device
        )
        self.mettre_a_jour_normalisation(obs.cpu().numpy())

        q = self.reseau(obs).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                actions_suiv = self.reseau(obs_suiv).argmax(dim=1, keepdim=True)
                q_suiv = self.reseau_cible(obs_suiv).gather(1, actions_suiv).squeeze(1)
            else:
                q_suiv = self.reseau_cible(obs_suiv).max(dim=1).values
            cibles = recompenses + self.gamma * q_suiv * (1.0 - finis)
        perte = self.perte(q, cibles)
        self.optimiseur.zero_grad()
        perte.backward()
        nn.utils.clip_grad_norm_(self.reseau.parameters(), 1.0)
        self.optimiseur.step()

        # Copie douce du réseau cible.
        with torch.no_grad():
            for p_cible, p in zip(
                self.reseau_cible.parameters(), self.reseau.parameters()
            ):
                p_cible.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return float(perte.item())

    def mettre_a_jour_epsilon(self, pas: int) -> None:
        """Décroît epsilon linéairement avec le nombre de pas.

        Args:
            pas: Nombre de pas d'environnement effectués.
        """
        self.eps = max(
            self.eps_fin,
            self.eps_fin + (1.0 - self.eps_fin) * max(0.0, 1.0 - pas / self.eps_decay),
        )


# ---------------------------------------------------------------------------
# Sauvegarde / chargement des champions (.npz compatible JEPA)
# ---------------------------------------------------------------------------
def sauvegarder_champion(
    agent: AgentDQN,
    chemin: Path,
    metriques: dict[str, float],
    extra: dict | None = None,
) -> Path:
    """Sauvegarde les poids du réseau en ``.npz`` (format flat p0..pN).

    Args:
        agent: Agent dont on sauvegarde le réseau.
        chemin: Chemin du fichier de sortie.
        metriques: Métriques embarquées.
        extra: Métadonnées additionnelles.

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
) -> ReseauQ:
    """Recharge un champion DQN depuis un ``.npz``.

    Args:
        chemin: Fichier ``.npz`` produit par ``sauvegarder_champion``.
        dim_obs: Dimension d'observation attendue.
        dim_action: Nombre d'actions.

    Returns:
        Réseau Q chargé (évaluation).

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
    reseau = ReseauQ(dim_obs, dim_action)
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
    agent: AgentDQN,
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
    """Exécute un run d'entraînement DQN complet.

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
        "DQN %s | %s | train=%d holdout=%d | device=%s",
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

    agent = AgentDQN(
        dim_obs=dim_obs,
        dim_action=3,
        lr=float(cfg.get("lr", 1e-3)),
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.005)),
        eps_debut=float(cfg.get("eps_debut", 1.0)),
        eps_fin=float(cfg.get("eps_fin", 0.02)),
        eps_decay=int(cfg.get("eps_decay", 4000)),
        double_dqn=bool(cfg.get("double_dqn", True)),
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
    apprendre_apres = int(cfg.get("apprendre_apres", 4))

    nb_champions = 0
    nb_records = 0
    meilleur_holdout = -np.inf
    meilleures_metriques: dict[str, float] = {}

    # --- Pré-remplissage du buffer (exploration pure) ---------------------
    obs = env.reset()
    for _ in range(prefill):
        action = int(np.random.randint(0, 3))
        obs_suiv, r, fini = env.step(action)
        buffer.ajouter(obs, action, r, obs_suiv, fini)
        obs = env.reset() if fini else obs_suiv

    for pas in range(1, steps + 1):
        action = agent.choisir_action(obs)
        obs_suiv, r, fini = env.step(action)
        buffer.ajouter(obs, action, r, obs_suiv, fini)
        agent.mettre_a_jour_epsilon(pas)

        perte = 0.0
        if pas % apprendre_apres == 0:
            for _ in range(apprendre_apres):
                perte = agent.apprendre(buffer, batch)

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
                "run_id": run_id, "pas": pas, "eps": round(agent.eps, 4),
                "perte": round(perte, 5),
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
                    {"algo": "dqn", "symbole": sym, "pas": pas},
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
                    "  ✗ pas %d : holdout np=%+.2f%% dd=%.2f%% (fitness=%.3f) eps=%.3f",
                    pas, m_h["net_profit"], m_h["max_drawdown"], m_h["fitness"], agent.eps,
                )

        if pas % (eval_holdout * 5) == 0:
            journal.info(
                "run %s pas %6d/%d | eps=%.3f | buffer=%d | champions=%d | %.0f pas/s",
                run_id, pas, steps, agent.eps, len(buffer), nb_champions,
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
        {"algo": "dqn", "symbole": sym, "pas": steps},
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
        "run_id": run_id, "symbole": sym, "algo": "dqn",
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
    """Test de validation : mini-run DQN sans crash (données synthétiques)."""
    journal.info("=== TEST RAPIDE DQN ===")
    np.random.seed(0)
    torch.manual_seed(0)
    nb = 1200
    prix = np.cumprod(1.0 + np.random.normal(0, 0.001, nb)) * 39000.0
    latents = np.random.normal(0, 1, (nb, DIM_LATENT)).astype(np.float32)
    env = EnvironnementTrading(prix, latents)
    agent = AgentDQN(dim_obs=env.dim_obs, dim_action=3)
    buffer = ReplayBuffer(5000)
    obs = env.reset()
    for pas in range(300):
        action = agent.choisir_action(obs)
        obs_suiv, r, fini = env.step(action)
        buffer.ajouter(obs, action, r, obs_suiv, fini)
        agent.mettre_a_jour_epsilon(pas)
        if pas % 4 == 0:
            agent.apprendre(buffer, 32)
        obs = env.reset() if fini else obs_suiv
    m = evaluer_politique(agent, prix, latents, nb_pas=200, nb_episodes=1)
    journal.info("np=%+.2f%% dd=%.2f%% fitness=%.3f", m["net_profit"], m["max_drawdown"], m["fitness"])
    chemin = sauvegarder_champion(
        agent, Path("/tmp/dqn_test_champion.npz"),
        {"fitness": 1.0, "net_profit": 0.5, "max_drawdown": 1.0, "sortino": 0.5,
         "win_rate": 50.0, "profit_factor": 1.2, "nb_trades": 10.0},
        {"algo": "dqn", "symbole": "TEST"},
    )
    reseau = charger_champion(chemin, dim_obs=env.dim_obs, dim_action=3)
    with torch.no_grad():
        x = torch.randn(1, env.dim_obs)
        assert reseau(x).shape == (1, 3)
    journal.info("✅ TEST DQN RÉUSSI — champion rechargé, inférence OK (%s)", chemin)


def main() -> None:
    """Point d'entrée CLI."""
    p = argparse.ArgumentParser(description="DQN E.V.A — RL trading M15 sur latents JEPA")
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
            "id": args.run_id or f"dqn_{args.symbole.replace('.', '_')}",
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
