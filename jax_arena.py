"""Arène décisionnelle E.V.A — module consolidé JAX (GPU 1).

Pipeline complet côté moteur décisionnel :
- pont DLPack zéro-copie PyTorch -> JAX (``bridge_pytorch_to_jax``),
- World Model latent récurrent (GRU compact en fonctions pures JAX),
- planificateur TD-MPC2 par méthode de l'entropie croisée (CEM, 5 000
  trajectoires simulées en parallèle via ``jax.vmap``),
- arène génétique ``JaxGeneticArena`` : évaluation et évolution de 64 agents
  parallélisées sous ``jax.jit`` + ``jax.vmap`` (crossover/mutation de
  Pytrees), fitness = (Sortino × 2) − MaxDrawdown + NetProfit.

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
Exécution du test de vitesse :
    PYTHONPATH=. venv/bin/python jax_arena.py
"""

from __future__ import annotations

import time
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import torch
from jax import Array

# ---------------------------------------------------------------------------
# Constantes globales (UPPER_CASE, PEP 8)
# ---------------------------------------------------------------------------

DIM_LATENT: int = 128
DIM_ACTION: int = 8
DIM_CACHE: int = 256
TAILLE_POPULATION: int = 64
NB_TRAJECTOIRES: int = 5000
NB_ELITES_CEM: int = 64
NB_ITERATIONS_CEM: int = 6
HORIZON_CEM: int = 5
GAMMA: float = 0.99
COUT_TRANSACTION: float = 0.0002
FENETRE_SORTINO: int = 256
CAPITAL_INITIAL: float = 100_000.0
EPS: float = 1e-8


# ---------------------------------------------------------------------------
# Consigne 2 — Pont DLPack zéro-copie PyTorch -> JAX
# ---------------------------------------------------------------------------


def bridge_pytorch_to_jax(
    torch_tensor: torch.Tensor,
    device_cible: object | None = None,
) -> Array:
    """Convertit un tenseur PyTorch en tableau JAX via DLPack zéro-copie.

    ``jax.dlpack.from_dlpack`` partage l'allocation VRAM existante quand JAX
    et PyTorch résident sur le même GPU (latence de transfert nulle) ;
    sinon XLA effectue un transfert pair-à-pair optimisé vers le device cible.

    Args:
        torch_tensor: Tenseur PyTorch (CPU ou CUDA) de forme ``(Batch, 128)``
            ou ``(128,)``, convertible en float32.
        device_cible: Device JAX de destination (ex. ``jax.devices()[1]``).
            Si ``None``, le device par défaut JAX est utilisé.

    Returns:
        Tableau JAX de forme identique, placé sur le device cible.

    Raises:
        TypeError: Si l'entrée n'est pas un ``torch.Tensor``.
        ValueError: Si le tenseur est vide ou a plus de 3 dimensions.
    """
    if not isinstance(torch_tensor, torch.Tensor):
        raise TypeError(
            f"torch.Tensor attendu, reçu {type(torch_tensor).__name__}"
        )
    if torch_tensor.ndim == 0 or torch_tensor.ndim > 3:
        raise ValueError(
            f"Tenseur 1D/2D/3D attendu, reçu {torch_tensor.ndim}D "
            f"de forme {tuple(torch_tensor.shape)}"
        )
    if torch_tensor.numel() == 0:
        raise ValueError("Tenseur vide — conversion refusée")
    tenseur = torch_tensor.detach().contiguous()
    if tenseur.dtype != torch.float32:
        tenseur = tenseur.float()
    tableau = jax.dlpack.from_dlpack(tenseur)
    if device_cible is not None:
        tableau = jax.device_put(tableau, device_cible)
    assert tableau.shape == tuple(tenseur.shape), (
        f"forme JAX {tableau.shape} != forme torch {tuple(tenseur.shape)}"
    )
    return tableau


# ---------------------------------------------------------------------------
# Consigne 3 — World Model latent (GRU compact en fonctions pures)
# ---------------------------------------------------------------------------


class ParametresWorldModel(NamedTuple):
    """Pytree des poids du World Model GRU + têtes récompense/valeur.

    Attributes:
        w_ih: Poids entrée->caché, ``(3*dim_cache, dim_latent+dim_action)``.
        w_hh: Poids caché->caché, ``(3*dim_cache, dim_cache)``.
        b_ih: Biais entrée, ``(3*dim_cache,)``.
        b_hh: Biais caché, ``(3*dim_cache,)``.
        w_recompense: Tête de récompense, ``(dim_cache, 1)``.
        w_valeur: Tête de valeur, ``(dim_cache, 1)``.
    """

    w_ih: Array
    w_hh: Array
    b_ih: Array
    b_hh: Array
    w_recompense: Array
    w_valeur: Array


def initialiser_world_model(
    cle: Array,
    dim_latent: int = DIM_LATENT,
    dim_action: int = DIM_ACTION,
    dim_cache: int = DIM_CACHE,
) -> ParametresWorldModel:
    """Initialise le GRU compact (Glorot uniforme) et les têtes scalaires.

    Args:
        cle: Clé PRNG JAX.
        dim_latent: Dimension de l'espace latent H (128 par spec).
        dim_action: Dimension de l'action virtuelle.
        dim_cache: Dimension cachée du GRU.

    Returns:
        ``ParametresWorldModel`` prêt pour ``appliquer_gru``.
    """
    k1, k2, k3, k4 = jax.random.split(cle, 4)
    dim_in = dim_latent + dim_action

    def _glorot(cle_l: Array, lignes: int, colonnes: int) -> Array:
        limite = jnp.sqrt(6.0 / (lignes + colonnes))
        return jax.random.uniform(
            cle_l, (lignes, colonnes), minval=-limite, maxval=limite
        )

    return ParametresWorldModel(
        w_ih=_glorot(k1, 3 * dim_cache, dim_in),
        w_hh=_glorot(k2, 3 * dim_cache, dim_cache),
        b_ih=jnp.zeros(3 * dim_cache),
        b_hh=jnp.zeros(3 * dim_cache),
        w_recompense=_glorot(k3, dim_cache, 1) * 0.1,
        w_valeur=_glorot(k4, dim_cache, 1) * 0.1,
    )


def appliquer_gru(
    params: ParametresWorldModel,
    latent: Array,
    action: Array,
    cache: Array,
) -> tuple[Array, Array, Array]:
    """Exécute un pas de transition du World Model : H_t × A_t -> H_{t+1}.

    GRU standard : r = σ(W_ir x + b_ir + W_hr h + b_hr),
    z = σ(W_iz x + b_iz + W_hz h + b_hz),
    n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn)),
    h' = (1 − z) ⊙ n + z ⊙ h.

    Args:
        params: Poids du modèle.
        latent: État latent H_t de forme ``(dim_latent,)``.
        action: Action virtuelle A_t de forme ``(dim_action,)``.
        cache: État caché GRU de forme ``(dim_cache,)``.

    Returns:
        Tuple ``(nouveau_cache, recompense, valeur)`` :
            - nouveau_cache : H_{t+1} de forme ``(dim_cache,)``,
            - recompense : scalaire prédit,
            - valeur : scalaire de bootstrap.
    """
    entree = jnp.concatenate([latent, action], axis=-1)
    gi = entree @ params.w_ih.T + params.b_ih
    gh = cache @ params.w_hh.T + params.b_hh
    i_r, i_z, i_n = jnp.split(gi, 3, axis=-1)
    h_r, h_z, h_n = jnp.split(gh, 3, axis=-1)
    r = jax.nn.sigmoid(i_r + h_r)
    z = jax.nn.sigmoid(i_z + h_z)
    n = jnp.tanh(i_n + r * h_n)
    nouveau_cache = (1.0 - z) * n + z * cache
    recompense = (nouveau_cache @ params.w_recompense).squeeze(-1)
    valeur = (nouveau_cache @ params.w_valeur).squeeze(-1)
    return nouveau_cache, recompense, valeur


def _simuler_trajectoire(
    params: ParametresWorldModel,
    latent_initial: Array,
    sequence_actions: Array,
    horizon: int,
    gamma: float,
) -> Array:
    """Calcule la récompense cumulée actualisée d'une séquence simulée.

    ``horizon`` et ``gamma`` sont capturés statiquement (closure) par
    l'appelant — jamais tracés par ``jit``.

    Args:
        params: Poids du World Model.
        latent_initial: État de départ de forme ``(dim_latent,)``.
        sequence_actions: Actions candidates ``(horizon, dim_action)``.
        horizon: Profondeur de planification (statique).
        gamma: Facteur d'actualisation (statique).

    Returns:
        Récompense totale actualisée (scalaire).
    """
    cache0 = jnp.zeros(params.b_hh.shape[0] // 3)
    facteurs = gamma ** jnp.arange(horizon, dtype=jnp.float32)

    def pas(cache: Array, entree: tuple[Array, Array]) -> tuple[Array, Array]:
        action, facteur = entree
        nouveau_cache, recompense, _ = appliquer_gru(
            params, cache[: latent_initial.shape[0]], action, cache
        )
        return nouveau_cache, recompense * facteur

    _, recompenses = jax.lax.scan(pas, cache0, (sequence_actions, facteurs))
    return jnp.sum(recompenses)


class TDMPC2Planner:
    """Planificateur MPC par entropie croisée (CEM) sur le World Model JAX.

    À chaque pas de temps, évalue ``nb_trajectoires`` séquences d'actions
    candidates en parallèle (``jax.vmap`` sous ``jax.jit``), sélectionne les
    élites, met à jour la distribution gaussienne d'échantillonnage, et
    retourne la première action de la meilleure séquence (MPC receding
    horizon).

    Attributes:
        params: Poids du World Model.
        dim_action: Dimension de l'action virtuelle.
        horizon: Profondeur de planification.
        nb_trajectoires: Trajectoires CEM par itération (5 000 par spec).
        nb_elites: Taille de l'élite CEM.
        nb_iterations: Itérations CEM par pas de temps.
        gamma: Facteur d'actualisation.
    """

    def __init__(
        self,
        params: ParametresWorldModel,
        dim_action: int = DIM_ACTION,
        horizon: int = HORIZON_CEM,
        nb_trajectoires: int = NB_TRAJECTOIRES,
        nb_elites: int = NB_ELITES_CEM,
        nb_iterations: int = NB_ITERATIONS_CEM,
        gamma: float = GAMMA,
    ) -> None:
        """Initialise le planificateur CEM et compile le simulateur batché.

        Args:
            params: Poids du World Model GRU.
            dim_action: Dimension de l'action virtuelle.
            horizon: Profondeur de planification.
            nb_trajectoires: Trajectoires CEM par itération (5 000 par spec).
            nb_elites: Taille de l'élite CEM.
            nb_iterations: Itérations CEM par pas de temps.
            gamma: Facteur d'actualisation.
        """
        self.params = params
        self.dim_action = dim_action
        self.horizon = horizon
        self.nb_trajectoires = nb_trajectoires
        self.nb_elites = nb_elites
        self.nb_iterations = nb_iterations
        self.gamma = gamma

        horizon_s, gamma_s = self.horizon, self.gamma

        def simuler_statique(
            p: ParametresWorldModel, latent: Array, actions: Array
        ) -> Array:
            return _simuler_trajectoire(p, latent, actions, horizon_s, gamma_s)

        self._simuler_batch = jax.jit(
            jax.vmap(simuler_statique, in_axes=(None, None, 0))
        )

    @partial(jax.jit, static_argnums=(0,))
    def _cem_pas(
        self,
        cle: Array,
        latent: Array,
        moyenne: Array,
        ecart: Array,
    ) -> tuple[Array, Array, Array]:
        """Exécute une itération CEM : échantillonne, évalue, met à jour.

        Args:
            cle: Clé PRNG.
            latent: État courant de forme ``(dim_latent,)``.
            moyenne: Moyenne courante ``(horizon, dim_action)``.
            ecart: Écart-type courant ``(horizon, dim_action)``.

        Returns:
            Tuple ``(nouvelle_moyenne, nouvel_ecart, meilleure_sequence)``.
        """
        bruit = jax.random.normal(
            cle, (self.nb_trajectoires, self.horizon, self.dim_action)
        )
        candidats = jnp.clip(moyenne[None] + ecart[None] * bruit, -1.0, 1.0)
        scores = self._simuler_batch(self.params, latent, candidats)
        idx_elites = jnp.argsort(scores)[-self.nb_elites :]
        elites = candidats[idx_elites]
        nouvelle_moyenne = jnp.mean(elites, axis=0)
        nouvel_ecart = jnp.std(elites, axis=0) + 1e-6
        return nouvelle_moyenne, nouvel_ecart, candidats[idx_elites[-1]]

    def planifier(
        self,
        cle: Array,
        latent: Array,
        moyenne_init: Array | None = None,
        ecart_init: float = 1.0,
    ) -> tuple[Array, Array]:
        """Planifie la meilleure action immédiate depuis l'état latent.

        Args:
            cle: Clé PRNG JAX.
            latent: État latent courant H_t de forme ``(dim_latent,)``.
            moyenne_init: Warm-start optionnel ``(horizon, dim_action)``.
            ecart_init: Écart-type initial si pas de warm-start.

        Returns:
            Tuple ``(action_immediate, moyenne_optimale)`` :
                - action_immediate : ``(dim_action,)`` première action,
                - moyenne_optimale : ``(horizon, dim_action)`` pour warm-start.

        Raises:
            ValueError: Si ``latent`` n'est pas 1D ou si ``moyenne_init`` a
                une forme incompatible.
        """
        if latent.ndim != 1:
            raise ValueError(f"latent 1D attendu, reçu {latent.ndim}D")
        if moyenne_init is not None and moyenne_init.shape != (
            self.horizon,
            self.dim_action,
        ):
            raise ValueError(
                f"moyenne_init {moyenne_init.shape} != "
                f"({self.horizon}, {self.dim_action})"
            )
        moyenne = (
            jnp.zeros((self.horizon, self.dim_action))
            if moyenne_init is None
            else moyenne_init
        )
        ecart = jnp.full((self.horizon, self.dim_action), ecart_init)
        meilleure = moyenne
        for _ in range(self.nb_iterations):
            cle, sous_cle = jax.random.split(cle)
            moyenne, ecart, meilleure = self._cem_pas(
                sous_cle, latent, moyenne, ecart
            )
        assert meilleure.shape == (self.horizon, self.dim_action), (
            f"meilleure séquence {meilleure.shape} != "
            f"({self.horizon}, {self.dim_action})"
        )
        return meilleure[0], moyenne


# ---------------------------------------------------------------------------
# Consignes 4 & 5 — Arène génétique JAX + fitness Sortino/Drawdown/NetProfit
# ---------------------------------------------------------------------------


class EtatSimulation(NamedTuple):
    """État scalaire d'un agent dans la simulation boursière.

    Attributes:
        cash: Equity courante.
        position: Position signée courante ∈ [-1, 1].
        prix_entree: Prix de la barre précédente (mark-to-market).
        equity_peak: Plus-haut historique de l'equity.
        historique_retours: Fenêtre glissante des retours ``(fenetre,)``.
        nb_trades: Nombre de trades fermés (changement de signe effectif).
        trades_gagnants: Nombre de trades fermés en profit.
        profit_brut: Somme des gains des trades gagnants.
        perte_brute: Somme des pertes des trades perdants (valeur positive).
        pnl_trade_courant: P&L accumulé du trade en cours.
    """

    cash: Array
    position: Array
    prix_entree: Array
    equity_peak: Array
    historique_retours: Array
    nb_trades: Array
    trades_gagnants: Array
    profit_brut: Array
    perte_brute: Array
    pnl_trade_courant: Array


class ResultatEvaluation(NamedTuple):
    """Métriques d'évaluation, vectorisées sur la population.

    Attributes:
        fitness: ``(pop,)`` = (Sortino × 2) − MaxDrawdown + NetProfit.
        net_profit: ``(pop,)`` profit net en % du capital initial.
        max_drawdown: ``(pop,)`` drawdown maximal en %.
        sortino: ``(pop,)`` ratio de Sortino.
        win_rate: ``(pop,)`` taux de trades gagnants en %.
        profit_factor: ``(pop,)`` profit_brut / perte_brute.
        nb_trades: ``(pop,)`` nombre de trades fermés.
    """

    fitness: Array
    net_profit: Array
    max_drawdown: Array
    sortino: Array
    win_rate: Array
    profit_factor: Array
    nb_trades: Array


def initialiser_population(
    cle: Array,
    taille_population: int = TAILLE_POPULATION,
    dim_latent: int = DIM_LATENT,
    dim_action: int = DIM_ACTION,
    dim_cache: int = DIM_CACHE,
    sigma_init: float = 0.02,
) -> ParametresWorldModel:
    """Génère une population de World Models par mutation d'un ancêtre.

    Args:
        cle: Clé PRNG.
        taille_population: Nombre d'agents (64 par spec).
        dim_latent: Dimension de l'espace latent.
        dim_action: Dimension de l'action virtuelle.
        dim_cache: Dimension cachée du GRU.
        sigma_init: Amplitude de la mutation initiale.

    Returns:
        ``ParametresWorldModel`` avec dimension population en tête,
        prêt pour ``jax.vmap``.
    """
    ancetre = initialiser_world_model(cle, dim_latent, dim_action, dim_cache)
    cles = jax.random.split(cle, taille_population)

    def muter(cle_i: Array) -> ParametresWorldModel:
        return jax.tree.map(
            lambda p: p + jax.random.normal(cle_i, p.shape) * sigma_init,
            ancetre,
        )

    return jax.vmap(muter)(cles)


def _pas_simulation(
    prix: Array,
    etat: EtatSimulation,
    action: Array,
) -> tuple[EtatSimulation, Array]:
    """Exécute un pas de trading pour UN agent (scalaire).

    Args:
        prix: Prix courant de l'actif (scalaire).
        etat: État de l'agent.
        action: Vecteur d'action ``(dim_action,)`` ; la première composante
            donne le signal de position via tanh.

    Returns:
        Tuple ``(nouvel_etat, retour_net)`` avec retour_net scalaire.
    """
    signal = jnp.tanh(action[0])
    variation = signal - etat.position
    cout = jnp.abs(variation) * COUT_TRANSACTION
    # Rendement par barre réaliste : position × variation relative du prix
    # depuis la barre PRÉCÉDENTE (mark-to-market) — rendements bornés.
    prix_precedent = jnp.maximum(etat.prix_entree, EPS)
    rendement_prix = (prix - prix_precedent) / prix_precedent
    retour_net = etat.position * rendement_prix - cout
    nouveau_cash = etat.cash * (1.0 + retour_net)
    historique = jnp.roll(etat.historique_retours, shift=-1)
    historique = historique.at[-1].set(retour_net)

    # --- Suivi des trades discrets -----------------------------------------
    # Un trade est "fermé" quand la position repasse par zéro ou change de
    # signe (passage long->short ou inverse). On accumule le P&L du trade.
    pnl_trade = etat.pnl_trade_courant + retour_net
    changement_signe = (
        (etat.position != 0.0)
        & (signal != 0.0)
        & (jnp.sign(signal) != jnp.sign(etat.position))
    )
    retour_neutre = (etat.position != 0.0) & (signal == 0.0)
    trade_ferme = changement_signe | retour_neutre

    nouveau_nb_trades = etat.nb_trades + trade_ferme.astype(jnp.float32)
    gagne = trade_ferme & (pnl_trade > 0.0)
    perd = trade_ferme & (pnl_trade <= 0.0)
    nouveau_gagnants = etat.trades_gagnants + gagne.astype(jnp.float32)
    nouveau_profit = etat.profit_brut + jnp.where(gagne, pnl_trade, 0.0)
    nouvelle_perte = etat.perte_brute + jnp.where(perd, -pnl_trade, 0.0)
    # Le P&L du trade repart à zéro à la fermeture, sinon s'accumule.
    nouveau_pnl_trade = jnp.where(trade_ferme, 0.0, pnl_trade)

    nouvel_etat = EtatSimulation(
        cash=nouveau_cash,
        position=signal,
        prix_entree=prix,  # mark-to-market : barre précédente
        equity_peak=jnp.maximum(etat.equity_peak, nouveau_cash),
        historique_retours=historique,
        nb_trades=nouveau_nb_trades,
        trades_gagnants=nouveau_gagnants,
        profit_brut=nouveau_profit,
        perte_brute=nouvelle_perte,
        pnl_trade_courant=nouveau_pnl_trade,
    )
    return nouvel_etat, retour_net


def _calculer_fitness(
    etat_final: EtatSimulation,
    capital_initial: float,
) -> ResultatEvaluation:
    """Calcule la fitness : (Sortino × 2) − MaxDrawdown + NetProfit.

    Le ratio de Sortino utilise la déviation des retours négatifs uniquement
    (downside deviation). Le drawdown est mesuré par rapport au plus-haut
    d'equity. Le profit net est exprimé en % du capital initial.

    Args:
        etat_final: État terminal d'un agent après simulation.
        capital_initial: Capital de départ (scalaire).

    Returns:
        ``ResultatEvaluation`` scalaire pour cet agent.
    """
    retours = etat_final.historique_retours
    moyenne = jnp.mean(retours)
    retours_neg = jnp.minimum(retours, 0.0)
    downside_std = jnp.sqrt(jnp.mean(retours_neg**2) + 1e-12)
    sortino = moyenne / downside_std
    drawdown = (etat_final.equity_peak - etat_final.cash) / jnp.maximum(
        etat_final.equity_peak, EPS
    )
    max_dd_pct = drawdown * 100.0
    net_profit = (etat_final.cash - capital_initial) / capital_initial * 100.0
    fitness = (sortino * 2.0) - max_dd_pct + net_profit
    # Trades discrets : win_rate et profit factor.
    nb_trades = etat_final.nb_trades
    win_rate = jnp.where(
        nb_trades > 0.0,
        etat_final.trades_gagnants / jnp.maximum(nb_trades, 1.0) * 100.0,
        0.0,
    )
    profit_factor = etat_final.profit_brut / jnp.maximum(
        etat_final.perte_brute, 1e-9
    )
    return ResultatEvaluation(
        fitness=fitness,
        net_profit=net_profit,
        max_drawdown=max_dd_pct,
        sortino=sortino,
        win_rate=win_rate,
        profit_factor=profit_factor,
        nb_trades=nb_trades,
    )


class JaxGeneticArena:
    """Incubateur darwinien : évalue et fait évoluer 64 agents sous vmap/jit.

    Évaluation parallélisée par ``jax.vmap`` (chaque agent trade la même
    séquence de marché avec ses propres poids), évolution compilée par
    ``jax.jit`` (sélection des champions, crossover arithmétique/géométrique
    des Pytrees, mutation gaussienne adaptive, garde anti-consanguinité par
    injection de bruit multiplicatif).

    Attributes:
        taille_population: Nombre d'agents (64 par spec).
        population: Pytree des poids, dimension population en tête.
    """

    def __init__(
        self,
        cle_maitre: Array,
        taille_population: int = TAILLE_POPULATION,
        dim_latent: int = DIM_LATENT,
        dim_action: int = DIM_ACTION,
        dim_cache: int = DIM_CACHE,
        capital_initial: float = CAPITAL_INITIAL,
        fenetre_sortino: int = FENETRE_SORTINO,
    ) -> None:
        """Initialise l'arène et génère la population initiale.

        Args:
            cle_maitre: Clé PRNG initiale.
            taille_population: Nombre d'agents (64 par spec).
            dim_latent: Dimension de l'espace latent H.
            dim_action: Dimension de l'action virtuelle.
            dim_cache: Dimension cachée du GRU.
            capital_initial: Capital de départ de la simulation.
            fenetre_sortino: Fenêtre de calcul du ratio de Sortino.
        """
        self.taille_population = taille_population
        self.dim_latent = dim_latent
        self.dim_action = dim_action
        self.dim_cache = dim_cache
        self.capital_initial = capital_initial
        self.fenetre_sortino = fenetre_sortino
        self.population = initialiser_population(
            cle_maitre, taille_population, dim_latent, dim_action, dim_cache
        )
        capital = self.capital_initial
        fenetre = self.fenetre_sortino

        def evaluer_un_agent(
            params_agent: ParametresWorldModel,
            donnees: tuple[Array, Array],
        ) -> ResultatEvaluation:
            """Simule un agent sur la séquence complète (fonction pure).

            Args:
                params_agent: Pytree des poids de l'agent.
                donnees: Tuple ``(prix_seq, latents_seq)`` du marché.

            Returns:
                ``ResultatEvaluation`` scalaire pour cet agent.
            """
            prix_seq, latents_seq = donnees
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            etat0 = EtatSimulation(
                cash=jnp.asarray(capital, dtype=jnp.float32),
                position=zero,
                # mark-to-market : démarre au premier prix du segment.
                prix_entree=prix_seq[0],
                equity_peak=jnp.asarray(capital, dtype=jnp.float32),
                historique_retours=jnp.zeros(fenetre, dtype=jnp.float32),
                nb_trades=zero,
                trades_gagnants=zero,
                profit_brut=zero,
                perte_brute=zero,
                pnl_trade_courant=zero,
            )

            def corps(
                etat: EtatSimulation, entree: tuple[Array, Array]
            ) -> tuple[EtatSimulation, Array]:
                prix, latent = entree
                cache = jnp.zeros(params_agent.b_hh.shape[0] // 3)
                action_sim = jnp.zeros(params_agent.w_ih.shape[1] - DIM_LATENT)
                _, _, valeur = appliquer_gru(
                    params_agent, latent, action_sim, cache
                )
                action = jnp.concatenate(
                    [jnp.tanh(valeur)[None], jnp.zeros(action_sim.shape[0] - 1)]
                )
                nouvel_etat, retour = _pas_simulation(prix, etat, action)
                return nouvel_etat, retour

            etat_final, _ = jax.lax.scan(
                corps, etat0, (prix_seq, latents_seq)
            )
            return _calculer_fitness(etat_final, capital)

        # Consigne 4 : vmap (population) + jit (compilation XLA).
        self._evaluer_population_jit = jax.jit(
            jax.vmap(evaluer_un_agent, in_axes=(0, None))
        )

    def evaluer_population(
        self, donnees_marche: Array, latents: Array
    ) -> ResultatEvaluation:
        """Évalue toute la population en parallèle via ``jax.vmap``.

        Args:
            donnees_marche: Prix de l'actif de forme ``(nb_pas,)``.
            latents: Encodages JEPA de forme ``(nb_pas, dim_latent)``.

        Returns:
            ``ResultatEvaluation`` vectorisé sur ``(taille_population,)``.

        Raises:
            ValueError: Si les formes d'entrée sont incompatibles.
        """
        if donnees_marche.ndim != 1:
            raise ValueError(
                f"donnees_marche 1D attendues, reçu {donnees_marche.ndim}D"
            )
        if latents.ndim != 2 or latents.shape[0] != donnees_marche.shape[0]:
            raise ValueError(
                f"latents (nb_pas, dim_latent) attendus, reçu {latents.shape}"
            )
        return self._evaluer_population_jit(
            self.population, (donnees_marche, latents)
        )

    @staticmethod
    def _crossover_pytree(
        parent_a: ParametresWorldModel,
        parent_b: ParametresWorldModel,
        cle: Array,
    ) -> ParametresWorldModel:
        """Croise deux Pytrees : moyenne arithmétique ou géométrique.

        Un masque binaire aléatoire par matrice choisit entre moyenne
        arithmétique (a+b)/2 et géométrique signée sign(a)·√|a·b|.

        Args:
            parent_a: Premier champion.
            parent_b: Second champion.
            cle: Clé PRNG du tirage.

        Returns:
            Pytree enfant.
        """
        cles = jax.random.split(cle, len(jax.tree.leaves(parent_a)))
        cles_iter = iter(cles)

        def melange(a: Array, b: Array) -> Array:
            masque = jax.random.bernoulli(next(cles_iter), 0.5, a.shape)
            arith = (a + b) * 0.5
            geo = jnp.sign(a) * jnp.sqrt(jnp.abs(a * b) + 1e-12)
            return jnp.where(masque, geo, arith)

        return jax.tree.map(melange, parent_a, parent_b)

    @staticmethod
    def _mutation_pytree(
        individu: ParametresWorldModel,
        cle: Array,
        taux: float,
        sigma: float,
    ) -> ParametresWorldModel:
        """Mute un Pytree par bruit gaussien adaptatif masqué.

        Args:
            individu: Pytree à muter.
            cle: Clé PRNG.
            taux: Probabilité de mutation par poids.
            sigma: Amplitude du bruit gaussien.

        Returns:
            Pytree muté.
        """
        feuilles, structure = jax.tree.flatten(individu)
        cles = jax.random.split(cle, len(feuilles))
        mutées = [
            f + jax.random.bernoulli(c, taux, f.shape) * (
                jax.random.normal(c, f.shape) * sigma
            )
            for f, c in zip(feuilles, cles)
        ]
        return jax.tree.unflatten(structure, mutées)

    def _diversite_population(self) -> Array:
        """Mesure la variance inter-poids moyenne de la population.

        Returns:
            Scalaire de diversité ; sous le seuil, l'anti-consanguinité
            déclenche un bruit multiplicatif.
        """
        aplati = jnp.concatenate(
            jax.tree.leaves(
                jax.tree.map(
                    lambda p: p.reshape(self.taille_population, -1),
                    self.population,
                )
            ),
            axis=1,
        )
        return jnp.mean(jnp.var(aplati, axis=0))

    @partial(jax.jit, static_argnums=(0, 4, 5, 6))
    def _evoluer_jit(
        self,
        population: ParametresWorldModel,
        fitness: Array,
        cle: Array,
        nb_elites: int,
        taux_mutation: float,
        sigma_mutation: float,
    ) -> ParametresWorldModel:
        """Produit la génération suivante (compilé XLA, population entière).

        Sélection des ``nb_elites`` champions, crossover par paires aléatoires
        de champions, mutation gaussienne — vectorisés via ``jax.vmap`` sur
        les Pytrees.

        Args:
            population: Pytree courant, dimension population en tête.
            fitness: Scores ``(taille_population,)``.
            cle: Clé PRNG de la génération.
            nb_elites: Champions conservés intacts.
            taux_mutation: Probabilité de mutation par poids.
            sigma_mutation: Amplitude de la mutation.

        Returns:
            Nouveau Pytree population.
        """
        idx_elites = jnp.argsort(fitness)[-nb_elites:]
        elites = jax.tree.map(lambda p: p[idx_elites], population)
        nb_remplissage = self.taille_population - nb_elites
        cles = jax.random.split(cle, nb_remplissage * 3)
        cles_a, cles_b, cles_m = (
            cles[:nb_remplissage],
            cles[nb_remplissage : 2 * nb_remplissage],
            cles[2 * nb_remplissage :],
        )
        idx_a = jax.random.randint(cles_a[0], (nb_remplissage,), 0, nb_elites)
        idx_b = jax.random.randint(cles_b[0], (nb_remplissage,), 0, nb_elites)
        parents_a = jax.tree.map(lambda p: p[idx_a], elites)
        parents_b = jax.tree.map(lambda p: p[idx_b], elites)
        enfants = jax.vmap(self._crossover_pytree)(parents_a, parents_b, cles_a)
        enfants_mutes = jax.vmap(
            lambda ind, c: self._mutation_pytree(ind, c, taux_mutation, sigma_mutation)
        )(enfants, cles_m)
        return jax.tree.map(
            lambda e, n: jnp.concatenate([e, n], axis=0), elites, enfants_mutes
        )

    def evoluer(
        self,
        fitness: Array,
        cle: Array,
        nb_elites: int = 16,
        taux_mutation: float = 0.1,
        sigma_mutation: float = 0.02,
        seuil_diversite: float = 1e-6,
        bruit_consanguinite: float = 1.5,
    ) -> None:
        """Exécute une génération complète : sélection, crossover, mutation.

        Args:
            fitness: Scores de la génération ``(taille_population,)``.
            cle: Clé PRNG de la génération.
            nb_elites: Nombre de champions conservés.
            taux_mutation: Probabilité de mutation par poids.
            sigma_mutation: Amplitude de la mutation gaussienne.
            seuil_diversite: Seuil de variance inter-poids déclenchant
                l'injection de bruit multiplicatif.
            bruit_consanguinite: Amplitude du bruit d'urgence.

        Raises:
            ValueError: Si ``fitness`` n'a pas la longueur de la population.
        """
        if fitness.ndim != 1 or fitness.shape[0] != self.taille_population:
            raise ValueError(
                f"fitness ({self.taille_population},) attendue, "
                f"reçu {fitness.shape}"
            )
        self.population = self._evoluer_jit(
            self.population,
            fitness,
            cle,
            nb_elites,
            taux_mutation,
            sigma_mutation,
        )
        if float(self._diversite_population()) < seuil_diversite:
            bruit = jax.tree.map(
                lambda p: jax.random.normal(cle, p.shape) * bruit_consanguinite,
                self.population,
            )
            self.population = jax.tree.map(
                lambda p, b: p * (1.0 + b), self.population, bruit
            )


# ---------------------------------------------------------------------------
# Consigne 6 — Test de vitesse d'inférence globale sur GPU
# ---------------------------------------------------------------------------


def tester_vitesse_inference() -> dict[str, float]:
    """Valide la vitesse d'inférence du pipeline complet sur GPU.

    Mesure : (1) le transfert DLPack PyTorch->JAX, (2) la planification CEM
    5 000 trajectoires, (3) l'évaluation vmap de 64 agents, (4) l'évolution
    génétique jittée. Le premier appel inclut la compilation XLA ; les
    mesures rapportées sont prises après warmup.

    Returns:
        Dictionnaire de latences en millisecondes et de débits.

    Raises:
        RuntimeError: Si aucun GPU JAX n'est disponible.
    """
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    if not gpus:
        raise RuntimeError("GPU JAX requis pour le test de vitesse")
    device_cible = gpus[-1]
    print(f"Device cible : {device_cible}")
    resultats: dict[str, float] = {}

    # 1) Pont DLPack : latent PyTorch GPU -> JAX.
    latent_torch = torch.randn(64, DIM_LATENT, device="cuda:0")
    _ = bridge_pytorch_to_jax(latent_torch, device_cible)  # warmup
    torch.cuda.synchronize()
    latents_jax = None
    t0 = time.perf_counter()
    for _ in range(20):
        latents_jax = bridge_pytorch_to_jax(latent_torch, device_cible)
    jax.block_until_ready(latents_jax)
    resultats["dlpack_ms"] = (time.perf_counter() - t0) / 20 * 1e3

    # 2) Planification CEM : 5 000 trajectoires × 6 itérations.
    cle = jax.random.PRNGKey(0)
    params = initialiser_world_model(cle)
    planner = TDMPC2Planner(params)  # défauts spec : 5000 traj, 6 iters
    latent_un = jax.random.normal(jax.random.PRNGKey(1), (DIM_LATENT,))
    _ = planner.planifier(jax.random.PRNGKey(2), latent_un)  # warmup/compile
    action = None
    t0 = time.perf_counter()
    for i in range(5):
        action, _ = planner.planifier(jax.random.PRNGKey(10 + i), latent_un)
    jax.block_until_ready(action)
    resultats["cem_5000_ms"] = (time.perf_counter() - t0) / 5 * 1e3

    # 3) Arène : évaluation vmap de 64 agents sur 512 pas de marché.
    nb_pas = 512
    arene = JaxGeneticArena(jax.random.PRNGKey(42))
    prix = jnp.cumprod(
        1.0 + jax.random.normal(jax.random.PRNGKey(3), (nb_pas,)) * 0.001
    ) * 2000.0
    latents_seq = jax.random.normal(
        jax.random.PRNGKey(4), (nb_pas, DIM_LATENT)
    )
    res = arene.evaluer_population(prix, latents_seq)  # warmup/compile
    jax.block_until_ready(res.fitness)
    t0 = time.perf_counter()
    for _ in range(5):
        res = arene.evaluer_population(prix, latents_seq)
    jax.block_until_ready(res.fitness)
    resultats["arena_eval_64_ms"] = (time.perf_counter() - t0) / 5 * 1e3

    # 4) Évolution génétique jittée (crossover + mutation des Pytrees).
    arene.evoluer(res.fitness, jax.random.PRNGKey(7))  # warmup/compile
    jax.block_until_ready(arene.population.w_ih)
    t0 = time.perf_counter()
    for i in range(5):
        arene.evoluer(res.fitness, jax.random.PRNGKey(20 + i))
    jax.block_until_ready(arene.population.w_ih)
    resultats["arena_evolution_ms"] = (time.perf_counter() - t0) / 5 * 1e3

    print("\n--- Vitesse d'inférence GPU (moyennes après warmup) ---")
    print(f"  DLPack (64×128)        : {resultats['dlpack_ms']:8.3f} ms")
    print(f"  CEM 5 000 trajectoires : {resultats['cem_5000_ms']:8.3f} ms")
    print(f"  Arène 64 agents (512p) : {resultats['arena_eval_64_ms']:8.3f} ms")
    print(f"  Évolution génétique    : {resultats['arena_evolution_ms']:8.3f} ms")
    return resultats


if __name__ == "__main__":
    tester_vitesse_inference()
