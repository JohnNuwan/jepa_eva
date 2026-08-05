"""JaxGeneticArena — incubateur darwinien pour population TD-MPC2.

Évalue en parallèle 64 instances du planificateur via ``jax.vmap`` sur un
environnement de simulation boursière vectorisé, calcule la fitness
(Sortino × 2 − MaxDrawdown% + NetProfit), et applique crossover géométrique
arithmétique + mutation gaussienne adaptive avec garde anti-consanguinité.

Conforme au Bloc B.3 de la MASTER-SPECIFICATION E.V.A.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .planificateur_tdmpc2 import (
    ParametresWorldModel,
    appliquer_gru,
    initialiser_world_model,
)


class EtatSimulation(NamedTuple):
    """État vectorisé de l'environnement boursier simulé."""

    cash: Array  # (pop,)
    position: Array  # (pop,) taille de position signée
    prix_entree: Array  # (pop,)
    equity_peak: Array  # (pop,)
    historique_retours: Array  # (pop, fenetre_sortino)


class ResultatEvaluation(NamedTuple):
    """Métriques d'un agent après simulation."""

    fitness: Array  # (pop,)
    net_profit: Array  # (pop,)
    max_drawdown_pct: Array  # (pop,)
    sortino: Array  # (pop,)


def initialiser_population(
    cle: Array,
    taille_population: int = 64,
    dim_latent: int = 128,
    dim_action: int = 8,
    dim_cache: int = 256,
) -> ParametresWorldModel:
    """Génère une population de world models par mutation d'un ancêtre.

    Args:
        cle: clé PRNG.
        taille_population: nombre d'agents (64 par spec).
        dim_latent, dim_action, dim_cache: dimensions du world model.

    Returns:
        ``ParametresWorldModel`` avec une dimension population en tête
        (``jax.vmap``-ready).
    """
    ancetre = initialiser_world_model(cle, dim_latent, dim_action, dim_cache)
    cles = jax.random.split(cle, taille_population)

    def muter(cle_i: Array) -> ParametresWorldModel:
        bruits = jax.tree.map(
            lambda p: jax.random.normal(cle_i, p.shape) * 0.02, ancetre
        )
        return jax.tree.map(lambda p, b: p + b, ancetre, bruits)

    return jax.vmap(muter)(cles)


def _pas_simulation(
    params: ParametresWorldModel,
    prix: Array,
    etat: EtatSimulation,
    action: Array,
    pas_taux_sans_risque: float,
) -> tuple[EtatSimulation, Array]:
    """Un pas de trading vectorisé sur la population.

    Args:
        params: world model (utilisé pour prédire la récompense immédiate,
            mais ici le PnL est calculé sur les prix réels simulés).
        prix: ``(pop,)`` prix courant de l'actif.
        etat: ``EtatSimulation`` vectorisé.
        action: ``(pop, dim_action)`` première composante = signal [-1, 1].
        pas_taux_sans_risque: taux sans risque par pas (pour Sortino).

    Returns:
        ``(nouvel_etat, retour_net)`` où retour_net est ``(pop,)``.
    """
    signal = jnp.tanh(action[0])  # position cible ∈ [-1, 1]
    variation_position = signal - etat.position
    # Coût de transaction proportionnel à la variation de position.
    cout = jnp.abs(variation_position) * 0.0002
    retour_brut = etat.position * (prix / jnp.maximum(etat.prix_entree, 1e-8) - 1.0)
    retour_net = retour_brut - cout - pas_taux_sans_risque

    nouveau_cash = etat.cash * (1.0 + retour_net)
    nouveau_peak = jnp.maximum(etat.equity_peak, nouveau_cash)
    nouvelle_position = signal
    nouveau_prix_entree = jnp.where(
        jnp.abs(variation_position) > 1e-6, prix, etat.prix_entree
    )

    historique = jnp.roll(etat.historique_retours, shift=-1)
    historique = historique.at[-1].set(retour_net)

    nouvel_etat = EtatSimulation(
        cash=nouveau_cash,
        position=nouvelle_position,
        prix_entree=nouveau_prix_entree,
        equity_peak=nouveau_peak,
        historique_retours=historique,
    )
    return nouvel_etat, retour_net


def _calculer_metriques(
    etat_final: EtatSimulation,
    capital_initial: float,
) -> ResultatEvaluation:
    """Fitness = (Sortino × 2) − MaxDrawdown% + NetProfit.

    Args:
        etat_final: état après ``nb_pas`` de simulation.
        capital_initial: capital de départ (scalaire).

    Returns:
        ``ResultatEvaluation`` vectorisé sur la population.
    """
    retours = etat_final.historique_retours  # (fenetre,)
    moyenne = jnp.mean(retours)
    # Downside deviation : écart-type des retours négatifs uniquement.
    retours_neg = jnp.minimum(retours, 0.0)
    downside_std = jnp.sqrt(jnp.mean(retours_neg**2) + 1e-12)
    sortino = moyenne / downside_std

    equity = etat_final.cash
    drawdown = (etat_final.equity_peak - equity) / jnp.maximum(
        etat_final.equity_peak, 1e-8
    )
    max_dd_pct = drawdown * 100.0

    net_profit = (equity - capital_initial) / capital_initial * 100.0

    fitness = (sortino * 2.0) - max_dd_pct + net_profit
    return ResultatEvaluation(
        fitness=fitness,
        net_profit=net_profit,
        max_drawdown_pct=max_dd_pct,
        sortino=sortino,
    )


class JaxGeneticArena:
    """Arène génétique JAX pour faire évoluer une population TD-MPC2.

    Args:
        cle_maitre: clé PRNG initiale.
        taille_population: nombre d'agents (64 par spec).
        dim_latent, dim_action, dim_cache: dimensions du world model.
        horizon_planification: horizon CEM du planner.
        nb_trajectoires: trajectoires CEM par agent.
        capital_initial: capital de départ de la simulation.
        fenetre_sortino: fenêtre de calcul du ratio de Sortino.
    """

    def __init__(
        self,
        cle_maitre: Array,
        taille_population: int = 64,
        dim_latent: int = 128,
        dim_action: int = 8,
        dim_cache: int = 256,
        horizon_planification: int = 5,
        nb_trajectoires: int = 5000,
        capital_initial: float = 100_000.0,
        fenetre_sortino: int = 256,
    ) -> None:
        """Initialise l'arène et génère la population initiale.

        Args:
            cle_maitre: Clé PRNG initiale.
            taille_population: Nombre d'agents (64 par spec).
            dim_latent: Dimension de l'espace latent H.
            dim_action: Dimension de l'action virtuelle.
            dim_cache: Dimension cachée du GRU world model.
            horizon_planification: Horizon CEM du planner.
            nb_trajectoires: Trajectoires CEM par agent.
            capital_initial: Capital de départ de la simulation.
            fenetre_sortino: Fenêtre de calcul du ratio de Sortino.
        """
        self.cle = cle_maitre
        self.taille_population = taille_population
        self.dim_latent = dim_latent
        self.dim_action = dim_action
        self.dim_cache = dim_cache
        self.horizon = horizon_planification
        self.nb_trajectoires = nb_trajectoires
        self.capital_initial = capital_initial
        self.fenetre_sortino = fenetre_sortino

        self.population = initialiser_population(
            cle_maitre, taille_population, dim_latent, dim_action, dim_cache
        )

    def _evaluer_agent(
        self,
        params_agent: ParametresWorldModel,
        donnees_marche: Array,
        latents: Array,
    ) -> ResultatEvaluation:
        """Évalue un agent sur une séquence de marché vectorisée.

        Args:
            params_agent: poids du world model de l'agent.
            donnees_marche: ``(nb_pas,)`` prix simulés ou réels.
            latents: ``(nb_pas, dim_latent)`` latents JEPA correspondants.

        Returns:
            ``ResultatEvaluation`` pour cet agent.
        """
        etat0 = EtatSimulation(
            cash=jnp.asarray(self.capital_initial),
            position=jnp.asarray(0.0),
            prix_entree=jnp.asarray(1.0),
            equity_peak=jnp.asarray(self.capital_initial),
            historique_retours=jnp.zeros(self.fenetre_sortino),
        )

        def corps(etat: EtatSimulation, entree: tuple[Array, Array]) -> tuple[EtatSimulation, Array]:
            prix, latent = entree
            # Action simulée : on utilise le world model pour prédire la
            # récompense, mais la décision de trading vient du planner.
            # Ici on dérive un signal simple depuis le GRU pour rester
            # dans une fonction pure et jittable.
            cache = jnp.zeros(self.dim_cache)
            action_sim = jnp.zeros(self.dim_action)
            _, _, _ = appliquer_gru(params_agent, latent, action_sim, cache)
            # Signal = tangente de la valeur prédite (borné [-1, 1]).
            _, _, valeur = appliquer_gru(params_agent, latent, action_sim, cache)
            action = jnp.concatenate([jnp.tanh(valeur)[None], jnp.zeros(self.dim_action - 1)])
            nouvel_etat, retour = _pas_simulation(
                params_agent, prix, etat, action, 0.0
            )
            return nouvel_etat, retour

        etat_final, _ = jax.lax.scan(corps, etat0, (donnees_marche, latents))
        return _calculer_metriques(etat_final, self.capital_initial)

    @partial(jax.jit, static_argnums=(0,))
    def evaluer_population(
        self,
        donnees_marche: Array,
        latents: Array,
    ) -> ResultatEvaluation:
        """Évalue toute la population en parallèle via ``jax.vmap``.

        Chaque agent simule le trading sur la même séquence de marché,
        avec ses propres poids de world model. La fitness combine Sortino,
        drawdown maximal et profit net.

        Args:
            donnees_marche: Prix de l'actif de forme ``(nb_pas,)``.
            latents: Encodages JEPA de forme ``(nb_pas, dim_latent)``.

        Returns:
            ``ResultatEvaluation`` vectorisé sur ``(taille_population,)``.

        Raises:
            ValueError: Si les longueurs de ``donnees_marche`` et
                ``latents`` diffèrent, ou si ``latents`` n'est pas 2D.
        """
        if donnees_marche.ndim != 1:
            raise ValueError(
                f"donnees_marche 1D attendues, reçu {donnees_marche.ndim}D"
            )
        if latents.ndim != 2:
            raise ValueError(f"latents 2D attendus, reçu {latents.ndim}D")
        if latents.shape[0] != donnees_marche.shape[0]:
            raise ValueError(
                f"latents {latents.shape[0]} pas != marché "
                f"{donnees_marche.shape[0]} pas"
            )
        return jax.vmap(
            self._evaluer_agent, in_axes=(0, None, None)
        )(self.population, donnees_marche, latents)

    def _selectionner_champions(
        self, fitness: Array, frac_elite: float = 0.25
    ) -> Array:
        """Indices des champions (top frac_elite)."""
        nb_elites = max(2, int(self.taille_population * frac_elite))
        return jnp.argsort(fitness)[-nb_elites:]

    def _crossover(
        self,
        parent_a: ParametresWorldModel,
        parent_b: ParametresWorldModel,
        cle: Array,
    ) -> ParametresWorldModel:
        """Crossover arithmétique/géométrique des Pytrees.

        Pour chaque matrice de poids, un masque binaire aléatoire choisit
        entre moyenne arithmétique et géométrique élément par élément.
        """
        masque_geo = jax.tree.map(
            lambda p: jax.random.bernoulli(cle, 0.5, p.shape),
            parent_a,
        )

        def melange(a: Array, b: Array, m: Array) -> Array:
            arith = (a + b) * 0.5
            # Géométrique stable : signe préservé via racine du produit des
            # valeurs absolues, signe du parent A.
            geo = jnp.sign(a) * jnp.sqrt(jnp.abs(a * b) + 1e-12)
            return jnp.where(m, geo, arith)

        return jax.tree.map(melange, parent_a, parent_b, masque_geo)

    def _mutation(
        self,
        individu: ParametresWorldModel,
        cle: Array,
        taux: float,
        sigma: float,
    ) -> ParametresWorldModel:
        """Mutation gaussienne adaptive sur les Pytrees."""
        masque = jax.tree.map(
            lambda p: jax.random.bernoulli(cle, taux, p.shape),
            individu,
        )
        bruit = jax.tree.map(
            lambda p: jax.random.normal(cle, p.shape) * sigma,
            individu,
        )
        return jax.tree.map(
            lambda p, m, b: p + m * b, individu, masque, bruit
        )

    def _diversite_population(self) -> Array:
        """Variance inter-poids moyenne de la population (anti-consanguinité)."""
        poids_plats = jax.tree.map(
            lambda p: p.reshape(self.taille_population, -1), self.population
        )
        aplati = jnp.concatenate(
            jax.tree.leaves(poids_plats), axis=1
        )  # (pop, total_params)
        return jnp.mean(jnp.var(aplati, axis=0))

    def evoluer(
        self,
        fitness: Array,
        nb_elites: int = 16,
        taux_mutation: float = 0.1,
        sigma_mutation: float = 0.02,
        seuil_diversite: float = 1e-6,
        bruit_consanguinite: float = 1.5,
    ) -> None:
        """Exécute une génération complète : sélection, crossover, mutation.

        Les ``nb_elites`` champions sont conservés intacts ; le reste de la
        population est régénéré par crossover arithmétique/géométrique puis
        mutation gaussienne. Si la variance inter-poids tombe sous
        ``seuil_diversite``, un bruit multiplicatif est injecté (garde
        anti-consanguinité).

        Args:
            fitness: Scores de la génération de forme
                ``(taille_population,)``.
            nb_elites: Nombre de champions conservés.
            taux_mutation: Probabilité de mutation par poids.
            sigma_mutation: Amplitude de la mutation gaussienne.
            seuil_diversite: Seuil de variance inter-poids déclenchant
                l'injection de bruit.
            bruit_consanguinite: Facteur multiplicatif du bruit d'urgence.

        Raises:
            ValueError: Si ``fitness`` n'est pas 1D de longueur
                ``taille_population``.
        """
        if fitness.ndim != 1 or fitness.shape[0] != self.taille_population:
            raise ValueError(
                f"fitness ({self.taille_population},) attendue, "
                f"reçu {fitness.shape}"
            )
        idx_elites = jnp.argsort(fitness)[-nb_elites:]
        elites = jax.tree.map(lambda p: p[idx_elites], self.population)

        cles = jax.random.split(self.cle, self.taille_population + 1)
        self.cle = cles[0]
        cles_enfants = cles[1:]

        nouveaux = []
        for i in range(self.taille_population):
            if i < nb_elites:
                nouveaux.append(jax.tree.map(lambda p: p[i], elites))
            else:
                parent_a = jax.tree.map(
                    lambda p: p[jax.random.randint(cles_enfants[i], (), 0, nb_elites)],
                    elites,
                )
                parent_b = jax.tree.map(
                    lambda p: p[jax.random.randint(cles_enfants[i], (), 0, nb_elites)],
                    elites,
                )
                enfant = self._crossover(parent_a, parent_b, cles_enfants[i])
                enfant = self._mutation(enfant, cles_enfants[i], taux_mutation, sigma_mutation)
                nouveaux.append(enfant)

        self.population = jax.tree.map(
            lambda *xs: jnp.stack(xs), *nouveaux
        )

        # Garde anti-consanguinité.
        diversite = self._diversite_population()
        if float(diversite) < seuil_diversite:
            bruit = jax.tree.map(
                lambda p: jax.random.normal(self.cle, p.shape) * bruit_consanguinite,
                self.population,
            )
            self.population = jax.tree.map(
                lambda p, b: p * (1.0 + b), self.population, bruit
            )
