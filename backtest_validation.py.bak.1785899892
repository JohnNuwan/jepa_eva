"""Backtest de validation E.V.A — évalue un champion sur le holdout.

Recharge les poids d'un champion sauvegardé (``registry_arena/champions/``)
et l'évalue sur la période JAMAIS vue à l'entraînement (barres > 80 % de
l'historique). C'est le test anti-overfitting : un champion qui généralise
doit rester rentable hors échantillon.

Usage :
    PYTHONPATH=. venv/bin/python backtest_validation.py --champion registry_arena/champions/champion_gen65.npz

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_arena import JaxGeneticArena, ParametresWorldModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.backtest")


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI.

    Returns:
        Espace de noms avec champion, symbole, timeframe et fraction holdout.
    """
    p = argparse.ArgumentParser(description="Backtest de validation holdout")
    p.add_argument("--champion", required=True, help="fichier champion_gen<N>.npz")
    p.add_argument("--symbole", default="XAUUSD")
    p.add_argument("--timeframe", default="m15")
    p.add_argument("--frac_entrainement", type=float, default=0.8)
    return p.parse_args()


def charger_champion(chemin: str | Path) -> ParametresWorldModel:
    """Recharge les poids Pytree d'un champion sauvegardé.

    Args:
        chemin: Chemin du fichier ``.npz`` produit par
            ``train_arena.sauvegarder_champion``.

    Returns:
        ``ParametresWorldModel`` du champion (sans dimension population).

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si la structure ne correspond pas au world model.
    """
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"Champion introuvable : {chemin}")
    donnees = np.load(chemin)
    # Reconstruit le Pytree dans l'ordre des champs du NamedTuple.
    gabarit = ParametresWorldModel(
        w_ih=None, w_hh=None, b_ih=None, b_hh=None,
        w_recompense=None, w_valeur=None,
    )
    feuilles = [jnp.asarray(donnees[f"p{i}"]) for i in range(6)]
    if len(feuilles) != 6:
        raise ValueError(f"6 feuilles attendues, reçu {len(feuilles)}")
    params = ParametresWorldModel(*feuilles)
    _ = gabarit  # documentation de la structure attendue
    return params


def backtester(args: argparse.Namespace) -> dict[str, float]:
    """Évalue le champion sur le holdout.

    Args:
        args: Arguments CLI.

    Returns:
        Métriques de validation (fitness, net_profit, drawdown, sortino,
        win_rate, profit_factor, nb_trades).

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
    debut_holdout = int(nb * args.frac_entrainement)

    prix_h = prix[debut_holdout:]
    latents_h = latents[debut_holdout:]
    journal.info(
        "Holdout : barres [%d:%d] (%d pas)", debut_holdout, nb, int(prix_h.shape[0])
    )

    params = charger_champion(args.champion)

    # Arène à population 1 : on injecte le champion seul.
    arene = JaxGeneticArena(jax.random.PRNGKey(0), taille_population=1)
    arene.population = jax.tree.map(lambda p: p[None], params)
    res = arene.evaluer_population(prix_h, latents_h)

    metriques = {
        "fitness": float(res.fitness[0]),
        "net_profit": float(res.net_profit[0]),
        "drawdown": float(res.max_drawdown[0]),
        "sortino": float(res.sortino[0]),
        "win_rate": float(res.win_rate[0]),
        "profit_factor": float(res.profit_factor[0]),
        "nb_trades": float(res.nb_trades[0]),
    }
    journal.info(
        "VALIDATION : fitness=%.3f np=%+.2f%% dd=%.2f%% sortino=%.3f "
        "wr=%.1f%% pf=%.2f trades=%d",
        metriques["fitness"], metriques["net_profit"], metriques["drawdown"],
        metriques["sortino"], metriques["win_rate"],
        metriques["profit_factor"], int(metriques["nb_trades"]),
    )
    # Verdict anti-overfitting.
    if metriques["net_profit"] > 0.0 and metriques["drawdown"] <= 5.0:
        journal.info("VERDICT : le champion GÉNÉRALISE (rentable hors échantillon)")
    else:
        journal.warning("VERDICT : SURAPPRENTISSAGE probable (non rentable holdout)")
    return metriques


def main() -> None:
    """Point d'entrée CLI."""
    backtester(parse_args())


if __name__ == "__main__":
    main()
