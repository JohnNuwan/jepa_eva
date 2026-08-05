"""Entraînement d'arène guidé par la généralisation (fitness train + holdout).

Variante avancée : l'évolution ne sélectionne plus les parents sur la seule
fitness train (qui dérive en surapprentissage), mais sur un score combiné
``fitness_train + λ × fitness_holdout`` évalué périodiquement. Les champions
sont promus uniquement s'ils généralisent. Cela stabilise l'évolution vers
des agents robustes au lieu d'une perle sur 40.

Usage :
    PYTHONPATH=. venv/bin/python train_arena_generalisee.py --symbole XAUUSD --generations 200

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_arena import JaxGeneticArena, ParametresWorldModel
from train_arena import ChampionRegistry, sauvegarder_champion
from train_arena_validated import evaluer_holdout, generalise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.train_generalisee")


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI.

    Returns:
        Espace de noms avec symbole, timeframe, générations et hyperparamètres.
    """
    p = argparse.ArgumentParser(description="Arène guidée par la généralisation")
    p.add_argument("--symbole", default="XAUUSD")
    p.add_argument("--timeframe", default="m15")
    p.add_argument("--generations", type=int, default=200)
    p.add_argument("--segment", type=int, default=512)
    p.add_argument("--taille_pop", type=int, default=64)
    p.add_argument("--nb_elites", type=int, default=16)
    p.add_argument("--frac_entrainement", type=float, default=0.8)
    p.add_argument("--lambda_holdout", type=float, default=1.0,
                   help="poids du score holdout dans la fitness d'évolution")
    p.add_argument("--eval_holdout", type=int, default=5,
                   help="rafraîchir le score holdout toutes les N gens")
    p.add_argument("--sortie", default="registry_arena_generalisee")
    return p.parse_args()


def evaluer_holdout_population(
    population: ParametresWorldModel,
    prix_h: jnp.ndarray,
    latents_h: jnp.ndarray,
) -> jnp.ndarray:
    """Évalue TOUTE la population sur le holdout (fitness par agent).

    Args:
        population: Pytree avec dimension population en tête.
        prix_h: Prix du holdout.
        latents_h: Latents du holdout.

    Returns:
        Fitness holdout ``(taille_population,)``.
    """
    taille = int(jax.tree.leaves(population)[0].shape[0])
    arene = JaxGeneticArena(jax.random.PRNGKey(0), taille_population=taille)
    arene.population = population
    res = arene.evaluer_population(prix_h, latents_h)
    return res.fitness


def entrainer(args: argparse.Namespace) -> None:
    """Boucle d'évolution guidée par la généralisation.

    Args:
        args: Arguments CLI.

    Raises:
        FileNotFoundError: Si les latents sont absents.
    """
    chemin_latents = Path("latents") / f"{args.symbole}_{args.timeframe}_latents.npz"
    if not chemin_latents.is_file():
        raise FileNotFoundError(f"Latents absents : {chemin_latents}")
    donnees = np.load(chemin_latents)
    prix = jnp.asarray(donnees["prix"], dtype=jnp.float32)
    latents = jnp.asarray(donnees["latents"], dtype=jnp.float32)
    nb = int(prix.shape[0])
    limite_train = int(nb * args.frac_entrainement)
    prix_h = prix[limite_train:]
    latents_h = latents[limite_train:]

    cle = jax.random.PRNGKey(0)
    arene = JaxGeneticArena(cle, taille_population=args.taille_pop)
    registry = ChampionRegistry(Path(args.sortie) / f"{args.symbole}_registry.jsonl")
    dossier_champions = Path(args.sortie) / "champions"

    journal.info(
        "Début : train<=%d | holdout=%d | pop=%d | %d gens | λ=%.1f | refresh 1/%d",
        limite_train, int(prix_h.shape[0]), args.taille_pop,
        args.generations, args.lambda_holdout, args.eval_holdout,
    )
    fitness_holdout = jnp.zeros(args.taille_pop)
    meilleur_holdout = -np.inf
    nb_promus = 0
    t0 = time.perf_counter()

    for gen in range(args.generations):
        debut = int(jax.random.randint(
            jax.random.PRNGKey(gen), (), 0, limite_train - args.segment
        ))
        prix_seg = jax.lax.dynamic_slice_in_dim(prix, debut, args.segment)
        latents_seg = jax.lax.dynamic_slice_in_dim(latents, debut, args.segment)

        res = arene.evaluer_population(prix_seg, latents_seg)

        # Rafraîchit le score holdout de toute la population périodiquement.
        if gen % args.eval_holdout == 0:
            fitness_holdout = evaluer_holdout_population(
                arene.population, prix_h, latents_h
            )

        # Score d'évolution combiné : train + λ × holdout.
        score_evolution = res.fitness + args.lambda_holdout * fitness_holdout

        idx_best = int(jnp.argmax(score_evolution))
        m_h = evaluer_holdout(
            jax.tree.map(lambda p: p[idx_best], arene.population), prix_h, latents_h
        )

        if generalise(m_h) and m_h["fitness"] > meilleur_holdout:
            meilleur_holdout = m_h["fitness"]
            nb_promus += 1
            registry.enregistrer(gen, {**m_h, "fitness_train": float(res.fitness[idx_best])})
            chemin = sauvegarder_champion(
                arene.population, idx_best, gen, dossier_champions, m_h
            )
            journal.info(
                "  ✓ gen %d : PROMU | holdout np=%+.2f%% dd=%.2f%% pf=%.2f "
                "trades=%d | score=%.3f -> %s",
                gen, m_h["net_profit"], m_h["drawdown"], m_h["profit_factor"],
                int(m_h["nb_trades"]), float(score_evolution[idx_best]), chemin.name,
            )

        # Évolution guidée par le score combiné (pas la fitness train seule).
        arene.evoluer(score_evolution, jax.random.PRNGKey(1000 + gen), nb_elites=args.nb_elites)

        if (gen + 1) % 20 == 0:
            journal.info(
                "gen %3d/%d | best_score=%.3f | holdout_max=%.3f | promus=%d | %.1f gen/s",
                gen + 1, args.generations, float(jnp.max(score_evolution)),
                meilleur_holdout, nb_promus,
                (gen + 1) / max(1e-9, time.perf_counter() - t0),
            )

    journal.info(
        "Terminé : record_holdout=%.3f | %d champions promus -> %s",
        meilleur_holdout, nb_promus, registry.chemin,
    )


def main() -> None:
    """Point d'entrée CLI."""
    entrainer(parse_args())


if __name__ == "__main__":
    main()
