"""Évaluation multi-symbole d'un champion E.V.A.

Teste la généralisation cross-symboles : un champion entraîné sur XAUUSD
est évalué sur le holdout de CHAQUE symbole. Un agent réellement robuste
doit rester rentable au-delà de son symbole d'entraînement.

Usage :
    PYTHONPATH=. venv/bin/python evaluer_multi_symbole.py \
        --champion registry_arena_generalisee/champions/champion_gen0.npz

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from backtest_validation import charger_champion
from train_arena_validated import evaluer_holdout, generalise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.multi")

SYMBOLES: tuple[str, ...] = (
    "XAUUSD", "EURUSD", "GBPUSD", "BTCUSD",
    "US30.cash", "US500.cash", "US100.cash", "GER40.cash",
)


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI.

    Returns:
        Espace de noms avec champion, timeframe et fraction holdout.
    """
    p = argparse.ArgumentParser(description="Évaluation multi-symbole")
    p.add_argument("--champion", required=True)
    p.add_argument("--timeframe", default="m15")
    p.add_argument("--frac_entrainement", type=float, default=0.8)
    return p.parse_args()


def evaluer(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Évalue le champion sur le holdout de chaque symbole.

    Args:
        args: Arguments CLI.

    Returns:
        Dictionnaire ``{symbole: métriques}``.

    Raises:
        FileNotFoundError: Si un fichier de latents est absent.
    """
    params = charger_champion(args.champion)
    journal.info("Champion : %s", args.champion)
    resultats: dict[str, dict[str, float]] = {}

    for symbole in SYMBOLES:
        chemin = Path("latents") / f"{symbole}_{args.timeframe}_latents.npz"
        if not chemin.is_file():
            journal.warning("Latents absents pour %s — ignoré", symbole)
            continue
        donnees = np.load(chemin)
        nb = int(donnees["prix"].shape[0])
        debut = int(nb * args.frac_entrainement)
        prix_h = jnp.asarray(donnees["prix"][debut:])
        latents_h = jnp.asarray(donnees["latents"][debut:])
        m = evaluer_holdout(params, prix_h, latents_h)
        resultats[symbole] = m
        verdict = "GÉNÉRALISE" if generalise(m) else "ne généralise pas"
        journal.info(
            "  %-12s np=%+8.2f%% dd=%6.2f%% wr=%5.1f%% pf=%6.2f trades=%3d | %s",
            symbole, m["net_profit"], m["drawdown"], m["win_rate"],
            m["profit_factor"], int(m["nb_trades"]), verdict,
        )

    nb_gen = sum(1 for m in resultats.values() if generalise(m))
    journal.info(
        "RÉSUMÉ : %d/%d symboles généralisés", nb_gen, len(resultats)
    )
    return resultats


def main() -> None:
    """Point d'entrée CLI."""
    evaluer(parse_args())


if __name__ == "__main__":
    main()
