"""Entraînement d'arène avec validation holdout — promotion conditionnelle.

Variante de ``train_arena.py`` : à chaque génération, le champion est évalué
sur le segment d'entraînement PUIS sur le holdout (barres > 80 %). Seuls les
champions qui GÉNÉRALISENT (rentables et drawdown maîtrisé hors échantillon)
sont promus et sauvegardés. C'est la boucle anti-overfitting complète.

Usage :
    PYTHONPATH=. venv/bin/python train_arena_validated.py --symbole XAUUSD --generations 100

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

from jax_arena import JaxGeneticArena
from train_arena import (
    ChampionRegistry,
    sauvegarder_champion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.train_validated")

# Seuils de généralisation (holdout).
DD_MAX_HOLDOUT: float = 5.0
NP_MIN_HOLDOUT: float = 0.0


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI.

    Returns:
        Espace de noms avec symbole, timeframe, générations et hyperparamètres.
    """
    p = argparse.ArgumentParser(description="Arène avec validation holdout")
    p.add_argument("--symbole", default="XAUUSD")
    p.add_argument("--timeframe", default="m15")
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--segment", type=int, default=512)
    p.add_argument("--taille_pop", type=int, default=64)
    p.add_argument("--nb_elites", type=int, default=16)
    p.add_argument("--frac_entrainement", type=float, default=0.8)
    p.add_argument("--eval_holdout", type=int, default=5,
                   help="évaluer le champion sur holdout toutes les N gens")
    p.add_argument("--sortie", default="registry_arena_validated")
    return p.parse_args()


def evaluer_holdout(
    params: object,
    prix_h: jnp.ndarray,
    latents_h: jnp.ndarray,
) -> dict[str, float]:
    """Évalue un champion unique sur le holdout.

    Args:
        params: Pytree du champion (sans dimension population).
        prix_h: Prix du holdout ``(nb_pas,)``.
        latents_h: Latents du holdout ``(nb_pas, 128)``.

    Returns:
        Métriques holdout (fitness, net_profit, drawdown, sortino, win_rate,
        profit_factor, nb_trades).
    """
    arene = JaxGeneticArena(jax.random.PRNGKey(0), taille_population=1)
    arene.population = jax.tree.map(lambda p: p[None], params)
    res = arene.evaluer_population(prix_h, latents_h)
    return {
        "fitness": float(res.fitness[0]),
        "net_profit": float(res.net_profit[0]),
        "drawdown": float(res.max_drawdown[0]),
        "sortino": float(res.sortino[0]),
        "win_rate": float(res.win_rate[0]),
        "profit_factor": float(res.profit_factor[0]),
        "nb_trades": float(res.nb_trades[0]),
    }


def generalise(m_h: dict[str, float]) -> bool:
    """Vérifie si le champion généralise sur le holdout.

    Args:
        m_h: Métriques holdout.

    Returns:
        ``True`` si rentable (net_profit > seuil) et drawdown maîtrisé.
    """
    return (
        m_h["net_profit"] > NP_MIN_HOLDOUT
        and m_h["drawdown"] <= DD_MAX_HOLDOUT
    )


def entrainer(args: argparse.Namespace) -> None:
    """Boucle train → backtest → promotion conditionnelle.

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
        "Début : train<=%d | holdout=%d pas | pop=%d | %d gens | validé 1/%d",
        limite_train, int(prix_h.shape[0]), args.taille_pop,
        args.generations, args.eval_holdout,
    )
    meilleur_fitness_train = -np.inf
    meilleur_fitness_holdout = -np.inf
    nb_generalises = 0
    t0 = time.perf_counter()

    for gen in range(args.generations):
        debut = int(jax.random.randint(
            jax.random.PRNGKey(gen), (), 0, limite_train - args.segment
        ))
        prix_seg = jax.lax.dynamic_slice_in_dim(prix, debut, args.segment)
        latents_seg = jax.lax.dynamic_slice_in_dim(latents, debut, args.segment)

        res = arene.evaluer_population(prix_seg, latents_seg)
        idx_best = int(jnp.argmax(res.fitness))
        fitness_best = float(res.fitness[idx_best])
        fitness_moy = float(jnp.mean(res.fitness))
        meilleur_fitness_train = max(meilleur_fitness_train, fitness_best)

        # Validation holdout périodique du champion courant.
        if (gen + 1) % args.eval_holdout == 0:
            params_best = jax.tree.map(lambda p: p[idx_best], arene.population)
            m_h = evaluer_holdout(params_best, prix_h, latents_h)
            if generalise(m_h) and m_h["fitness"] > meilleur_fitness_holdout:
                meilleur_fitness_holdout = m_h["fitness"]
                nb_generalises += 1
                registry.enregistrer(gen, {**m_h, "fitness_train": fitness_best})
                chemin = sauvegarder_champion(
                    arene.population, idx_best, gen, dossier_champions, m_h
                )
                journal.info(
                    "  ✓ gen %d : GÉNÉRALISE | holdout np=%+.2f%% dd=%.2f%% "
                    "wr=%.1f%% pf=%.2f | train=%.3f -> %s",
                    gen, m_h["net_profit"], m_h["drawdown"], m_h["win_rate"],
                    m_h["profit_factor"], fitness_best, chemin.name,
                )
            else:
                journal.info(
                    "  ✗ gen %d : holdout np=%+.2f%% dd=%.2f%% (pas de généralisation)",
                    gen, m_h["net_profit"], m_h["drawdown"],
                )

        arene.evoluer(res.fitness, jax.random.PRNGKey(1000 + gen), nb_elites=args.nb_elites)

        if (gen + 1) % 10 == 0:
            journal.info(
                "gen %3d/%d | best_train=%.3f | moy=%.3f | record_holdout=%.3f "
                "| généralisés=%d | %.1f gen/s",
                gen + 1, args.generations, fitness_best, fitness_moy,
                meilleur_fitness_holdout, nb_generalises,
                (gen + 1) / max(1e-9, time.perf_counter() - t0),
            )

    journal.info(
        "Terminé : record_train=%.3f | record_holdout=%.3f | %d champions "
        "généralisés -> %s",
        meilleur_fitness_train, meilleur_fitness_holdout, nb_generalises,
        registry.chemin,
    )


def main() -> None:
    """Point d'entrée CLI."""
    entrainer(parse_args())


if __name__ == "__main__":
    main()
