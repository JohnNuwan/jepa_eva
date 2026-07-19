"""TDMPC2Planner — planification MPC dans l'espace latent JEPA.

World model récurrent (GRU compact JAX) simulant les transitions
H_t -> H_{t+1} sous l'effet d'actions virtuelles A_t, combiné à une méthode
de l'entropie croisée (CEM) qui évalue 5 000 trajectoires d'actions futures
en parallèle via ``jax.vmap`` pour extraire la séquence maximisant la
récompense attendue.

Conforme au Bloc B.2 de la MASTER-SPECIFICATION E.V.A.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class EtatGRU(NamedTuple):
    """Pytree d'état du GRU world model."""

    cache: Array  # (batch, dim_cache)


class ParametresWorldModel(NamedTuple):
    """Pytree des poids du world model TD-MPC2."""

    w_ih: Array  # (3*dim_cache, dim_latent + dim_action)
    w_hh: Array  # (3*dim_cache, dim_cache)
    b_ih: Array  # (3*dim_cache,)
    b_hh: Array  # (3*dim_cache,)
    w_recompense: Array  # (dim_cache, 1)
    w_valeur: Array  # (dim_cache, 1)


def initialiser_world_model(
    cle: Array,
    dim_latent: int = 128,
    dim_action: int = 8,
    dim_cache: int = 256,
) -> ParametresWorldModel:
    """Initialisation orthogonal/Xavier du GRU compact et des têtes.

    Args:
        cle: clé PRNG JAX.
        dim_latent: dimension de l'espace latent H (128 par spec).
        dim_action: dimension de l'action virtuelle continue.
        dim_cache: dimension cachée du GRU.

    Returns:
        ``ParametresWorldModel`` prêt pour ``appliquer_gru``.
    """
    k1, k2, k3, k4 = jax.random.split(cle, 4)
    dim_in = dim_latent + dim_action

    def _glorot(cle: Array, lignes: int, colonnes: int) -> Array:
        limite = jnp.sqrt(6.0 / (lignes + colonnes))
        return jax.random.uniform(cle, (lignes, colonnes), minval=-limite, maxval=limite)

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
    """Un pas de transition du world model GRU.

    Args:
        params: poids du modèle.
        latent: ``(batch, dim_latent)`` état latent H_t.
        action: ``(batch, dim_action)`` action virtuelle A_t.
        cache: ``(batch, dim_cache)`` état caché GRU.

    Returns:
        Tuple ``(nouveau_cache, recompense_predite, valeur_predite)`` :
            - nouveau_cache : ``(batch, dim_cache)`` H_{t+1},
            - recompense_predite : ``(batch,)`` scalaire par échantillon,
            - valeur_predite : ``(batch,)`` bootstrap value.
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
    """Récompense cumulée actualisée d'une séquence d'actions simulée.

    ``horizon`` est capturé statiquement (closure) — jamais tracé par jit.
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
    """Planificateur MPC par entropie croisée (CEM) sur le world model JAX.

    À chaque pas de temps, évalue ``nb_trajectoires`` séquences d'actions
    candidates en parallèle (``jax.vmap``), sélectionne les élites, met à jour
    la distribution gaussienne d'échantillonnage, et retourne la première
    action de la meilleure séquence.

    Args:
        params: poids du world model.
        dim_action: dimension de l'action virtuelle.
        horizon: profondeur de planification (défaut 5).
        nb_trajectoires: trajectoires CEM par itération (5 000 par spec).
        nb_elites: taille de l'élite CEM.
        nb_iterations: itérations CEM par pas de temps.
        gamma: facteur d'actualisation.
    """

    def __init__(
        self,
        params: ParametresWorldModel,
        dim_action: int = 8,
        horizon: int = 5,
        nb_trajectoires: int = 5000,
        nb_elites: int = 64,
        nb_iterations: int = 6,
        gamma: float = 0.99,
    ) -> None:
        """Initialise le planificateur CEM.

        Args:
            params: Poids du world model GRU.
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

        # horizon et gamma sont capturés statiquement via closure.
        horizon, gamma = self.horizon, self.gamma

        def simuler_statique(
            params: ParametresWorldModel,
            latent: Array,
            actions: Array,
        ) -> Array:
            return _simuler_trajectoire(params, latent, actions, horizon, gamma)

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
        """Une itération CEM : échantillonnage, évaluation, mise à jour.

        Args:
            cle: clé PRNG.
            latent: ``(dim_latent,)`` état courant.
            moyenne: ``(horizon, dim_action)`` moyenne courante.
            ecart: ``(horizon, dim_action)`` écart-type courant.

        Returns:
            ``(moyenne, ecart, meilleure_sequence)``.
        """
        bruit = jax.random.normal(
            cle, (self.nb_trajectoires, self.horizon, self.dim_action)
        )
        candidats = moyenne[None] + ecart[None] * bruit
        candidats = jnp.clip(candidats, -1.0, 1.0)

        scores = self._simuler_batch(self.params, latent, candidats)
        idx_elites = jnp.argsort(scores)[-self.nb_elites :]
        elites = candidats[idx_elites]

        nouvelle_moyenne = jnp.mean(elites, axis=0)
        nouvel_ecart = jnp.std(elites, axis=0) + 1e-6
        meilleure = candidats[idx_elites[-1]]
        return nouvelle_moyenne, nouvel_ecart, meilleure

    def planifier(
        self,
        cle: Array,
        latent: Array,
        moyenne_init: Array | None = None,
        ecart_init: float = 1.0,
    ) -> tuple[Array, Array]:
        """Planifie la meilleure action immédiate depuis l'état latent.

        Exécute ``nb_iterations`` itérations CEM : échantillonnage de
        ``nb_trajectoires`` séquences d'actions, évaluation parallèle via
        ``vmap``, sélection des élites, mise à jour de la distribution
        gaussienne. Retourne la première action de la meilleure séquence.

        Args:
            cle: Clé PRNG JAX.
            latent: État latent courant H_t de forme ``(dim_latent,)``.
            moyenne_init: Warm-start optionnel de forme
                ``(horizon, dim_action)``.
            ecart_init: Écart-type initial si pas de warm-start.

        Returns:
            Tuple ``(action_immediate, moyenne_optimale)`` :
                - action_immediate : ``(dim_action,)`` première action,
                - moyenne_optimale : ``(horizon, dim_action)`` pour warm-start
                  au pas de temps suivant.

        Raises:
            ValueError: Si ``latent`` n'est pas 1D de dimension ``dim_latent``,
                ou si ``moyenne_init`` a une forme incompatible.
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
        if moyenne_init is None:
            moyenne = jnp.zeros((self.horizon, self.dim_action))
        else:
            moyenne = moyenne_init
        ecart = jnp.full((self.horizon, self.dim_action), ecart_init)

        meilleure = moyenne
        for i in range(self.nb_iterations):
            cle, sous_cle = jax.random.split(cle)
            moyenne, ecart, meilleure = self._cem_pas(
                sous_cle, latent, moyenne, ecart
            )
        assert meilleure.shape == (self.horizon, self.dim_action), (
            f"meilleure séquence {meilleure.shape} != "
            f"({self.horizon}, {self.dim_action})"
        )
        return meilleure[0], moyenne
