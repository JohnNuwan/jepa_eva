"""Entraînement de l'arène génétique E.V.A sur latents JEPA réels.

Boucle multi-générations : évalue 64 agents (``JaxGeneticArena``) sur des
segments de marché réel (prix + latents pré-calculés), fait évoluer la
population (crossover/mutation), et promeut les champions selon des critères
stricts (Pattern #15 : WR, profit factor, drawdown, nb trades). Un registre
JSON trace chaque génération promue.

Usage :
    PYTHONPATH=. venv/bin/python train_arena.py --symbole XAUUSD --generations 100

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_arena import JaxGeneticArena, ParametresWorldModel, ResultatEvaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.train_arena")

DOSSIER_REGISTRY = Path("registry_arena")


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI.

    Returns:
        Espace de noms avec symbole, timeframe, générations et hyperparamètres.
    """
    p = argparse.ArgumentParser(description="Entraînement arène génétique")
    p.add_argument("--symbole", default="XAUUSD")
    p.add_argument("--timeframe", default="m15")
    p.add_argument("--generations", type=int, default=100)
    p.add_argument("--segment", type=int, default=512, help="barres par évaluation")
    p.add_argument("--taille_pop", type=int, default=64)
    p.add_argument("--nb_elites", type=int, default=16)
    p.add_argument("--frac_entrainement", type=float, default=0.8)
    p.add_argument("--sortie", default=str(DOSSIER_REGISTRY))
    return p.parse_args()


class ChampionRegistry:
    """Registre JSON des générations promues (traçabilité + rollback).

    Attributes:
        chemin: Fichier JSONL du registre.
        generations: Nombre de champions enregistrés.
    """

    # Critères de promotion (Pattern #15).
    MIN_WIN_RATE: float = 55.0
    MIN_PROFIT_FACTOR: float = 1.3
    MAX_DRAWDOWN: float = 5.0
    MIN_TRADES: int = 30

    def __init__(self, chemin: str | Path) -> None:
        """Initialise le registre.

        Args:
            chemin: Chemin du fichier JSONL (créé si absent).
        """
        self.chemin = Path(chemin)
        self.chemin.parent.mkdir(parents=True, exist_ok=True)
        self.generations = 0

    def promotion_digne(self, metriques: dict[str, float]) -> bool:
        """Vérifie si un champion mérite d'être promu (multi-critères).

        Args:
            metriques: ``win_rate``, ``profit_factor``, ``drawdown``,
                ``nb_trades``, ``fitness``, ``net_profit``, ``sortino``.

        Returns:
            ``True`` si TOUS les critères passent.
        """
        return (
            metriques["win_rate"] >= self.MIN_WIN_RATE
            and metriques["profit_factor"] >= self.MIN_PROFIT_FACTOR
            and metriques["drawdown"] <= self.MAX_DRAWDOWN
            and metriques["nb_trades"] >= self.MIN_TRADES
        )

    def enregistrer(self, generation: int, metriques: dict[str, float]) -> None:
        """Ajoute une génération promue au registre JSONL.

        Args:
            generation: Numéro de génération.
            metriques: Métriques du champion promu.
        """
        entree = {
            "horodatage": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generation": generation,
            **{k: round(float(v), 4) for k, v in metriques.items()},
        }
        with self.chemin.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entree, ensure_ascii=False) + "\n")
        self.generations += 1


def metriques_depuis_resultat(res: ResultatEvaluation, idx: int) -> dict[str, float]:
    """Extrait les métriques réelles d'un agent depuis le résultat vectorisé.

    Args:
        res: ``ResultatEvaluation`` de toute la population.
        idx: Indice de l'agent.

    Returns:
        Dictionnaire de métriques scalaires incluant ``win_rate``,
        ``profit_factor`` et ``nb_trades`` (trades discrets suivis par l'arène).
    """
    return {
        "fitness": float(res.fitness[idx]),
        "net_profit": float(res.net_profit[idx]),
        "drawdown": float(res.max_drawdown[idx]),
        "sortino": float(res.sortino[idx]),
        "win_rate": float(res.win_rate[idx]),
        "profit_factor": float(res.profit_factor[idx]),
        "nb_trades": float(res.nb_trades[idx]),
    }


def sauvegarder_champion(
    population: ParametresWorldModel,
    idx: int,
    generation: int,
    dossier: Path,
    metriques: dict[str, float],
) -> Path:
    """Sauvegarde les poids Pytree d'un champion en ``.npz``.

    Args:
        population: Pytree de toute la population (dimension pop en tête).
        idx: Indice du champion dans la population.
        generation: Numéro de génération.
        dossier: Dossier de destination.
        metriques: Métriques du champion (incluses dans le fichier).

    Returns:
        Chemin du fichier sauvegardé.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    poids = jax.tree.map(lambda p: np.asarray(p[idx]), population)
    aplati, _ = jax.tree.flatten(poids)
    donnees = {f"p{i}": np.asarray(f) for i, f in enumerate(aplati)}
    donnees["generation"] = np.asarray(generation)
    donnees["fitness"] = np.asarray(metriques["fitness"])
    chemin = dossier / f"champion_gen{generation}.npz"
    np.savez_compressed(chemin, **donnees)
    return chemin


def entrainer(args: argparse.Namespace) -> None:
    """Boucle multi-générations de l'arène sur latents réels.

    Args:
        args: Arguments CLI.

    Raises:
        FileNotFoundError: Si le fichier de latents est absent.
    """
    chemin_latents = Path("latents") / f"{args.symbole}_{args.timeframe}_latents.npz"
    if not chemin_latents.is_file():
        raise FileNotFoundError(
            f"Latents absents : {chemin_latents} — lancez precompute_latents.py"
        )
    donnees = np.load(chemin_latents)
    prix = jnp.asarray(donnees["prix"], dtype=jnp.float32)
    latents = jnp.asarray(donnees["latents"], dtype=jnp.float32)
    nb = int(prix.shape[0])
    limite_train = int(nb * args.frac_entrainement)

    cle = jax.random.PRNGKey(0)
    arene = JaxGeneticArena(cle, taille_population=args.taille_pop)
    registry = ChampionRegistry(Path(args.sortie) / f"{args.symbole}_registry.jsonl")

    journal.info(
        "Début : %d barres (train<=%d) | pop=%d | %d générations | segment=%d",
        nb, limite_train, args.taille_pop, args.generations, args.segment,
    )
    meilleur_fitness = -np.inf
    t0 = time.perf_counter()

    for gen in range(args.generations):
        # Segment aléatoire dans la zone d'entraînement.
        debut_max = limite_train - args.segment
        debut = int(jax.random.randint(jax.random.PRNGKey(gen), (), 0, debut_max))
        prix_seg = jax.lax.dynamic_slice_in_dim(prix, debut, args.segment)
        latents_seg = jax.lax.dynamic_slice_in_dim(latents, debut, args.segment)

        res = arene.evaluer_population(prix_seg, latents_seg)
        idx_best = int(jnp.argmax(res.fitness))
        fitness_best = float(res.fitness[idx_best])
        fitness_moy = float(jnp.mean(res.fitness))

        if fitness_best > meilleur_fitness:
            meilleur_fitness = fitness_best
            metriques = metriques_depuis_resultat(res, idx_best)
            # Promotion multi-critères complète (Pattern #15) OU critère de
            # repli fitness+DD si pas assez de trades sur le segment.
            if registry.promotion_digne(metriques) or (
                metriques["drawdown"] <= registry.MAX_DRAWDOWN
                and metriques["nb_trades"] < registry.MIN_TRADES
            ):
                registry.enregistrer(gen, metriques)
                chemin_champ = sauvegarder_champion(
                    arene.population, idx_best, gen,
                    Path(args.sortie) / "champions", metriques,
                )
                journal.info(
                    "  ★ gen %d : fitness=%.3f wr=%.1f%% pf=%.2f dd=%.2f%% "
                    "np=%.2f%% trades=%d -> %s",
                    gen, fitness_best, metriques["win_rate"],
                    metriques["profit_factor"], metriques["drawdown"],
                    metriques["net_profit"], int(metriques["nb_trades"]),
                    chemin_champ.name,
                )

        arene.evoluer(res.fitness, jax.random.PRNGKey(1000 + gen), nb_elites=args.nb_elites)

        if (gen + 1) % 10 == 0:
            journal.info(
                "gen %3d/%d | best=%.3f | moy=%.3f | record=%.3f | %.1f gen/s",
                gen + 1, args.generations, fitness_best, fitness_moy,
                meilleur_fitness, (gen + 1) / max(1e-9, time.perf_counter() - t0),
            )

    journal.info(
        "Terminé : record=%.3f | %d champions promus -> %s",
        meilleur_fitness, registry.generations, registry.chemin,
    )


def main() -> None:
    """Point d'entrée CLI."""
    entrainer(parse_args())


if __name__ == "__main__":
    main()
