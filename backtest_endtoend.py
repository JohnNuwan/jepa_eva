"""Backtest end-to-end — rejeu historique de la chaîne complète E.V.A.

Rejoue l'historique réel barre par barre : latents pré-calculés → pont DLPack
→ planification CEM (world model entraîné si disponible) → sanitizer 1 % →
disjoncteur 4 % → simulation d'exécution avec P&L réel. Produit les métriques
de performance complètes de la stratégie end-to-end.

Usage :
    PYTHONPATH=. venv/bin/python backtest_endtoend.py \
        --champion registry_arena_generalisee/champions/champion_gen0.npz

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

from action_sanitizer import ActionSanitizer, DrawdownDisconnector
from backtest_validation import charger_champion
from jax_arena import (
    ParametresWorldModel,
    TDMPC2Planner,
    bridge_pytorch_to_jax,
    initialiser_world_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.endtoend")

EQUITY_INITIAL: float = 100_000.0
DISTANCE_SL: float = 5.0
CONTRAT: float = 100.0  # oz par lot (XAUUSD)


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI.

    Returns:
        Espace de noms avec champion, symbole, timeframe et bornes de rejeu.
    """
    p = argparse.ArgumentParser(description="Backtest end-to-end E.V.A")
    p.add_argument("--champion", default=None,
                   help="champion npz (optionnel — sinon world model aléatoire)")
    p.add_argument("--world_model", default=None,
                   help="world model entraîné npz (optionnel)")
    p.add_argument("--symbole", default="XAUUSD")
    p.add_argument("--timeframe", default="m15")
    p.add_argument("--frac_entrainement", type=float, default=0.8)
    p.add_argument("--nb_pas", type=int, default=2000, help="barres à rejouer")
    return p.parse_args()


def charger_world_model(chemin: str | Path) -> ParametresWorldModel:
    """Recharge un world model entraîné (6 feuilles).

    Args:
        chemin: Chemin du fichier npz produit par ``train_world_model.py``.

    Returns:
        ``ParametresWorldModel``.
    """
    donnees = np.load(chemin)
    feuilles = [jnp.asarray(donnees[f"p{i}"]) for i in range(6)]
    return ParametresWorldModel(*feuilles)


def backtester(args: argparse.Namespace) -> dict[str, float]:
    """Rejoue l'historique et calcule les métriques end-to-end.

    Args:
        args: Arguments CLI.

    Returns:
        Métriques : pnl_total_pct, nb_ordres, nb_refus, nb_disjonctions,
        win_rate, equity_finale.

    Raises:
        FileNotFoundError: Si les latents sont absents.
    """
    chemin_latents = Path("latents") / f"{args.symbole}_{args.timeframe}_latents.npz"
    if not chemin_latents.is_file():
        raise FileNotFoundError(f"Latents absents : {chemin_latents}")
    donnees = np.load(chemin_latents)
    prix = np.asarray(donnees["prix"], dtype=np.float32)
    latents = np.asarray(donnees["latents"], dtype=np.float32)
    nb = int(prix.shape[0])
    debut = int(nb * args.frac_entrainement)
    fin = min(nb, debut + args.nb_pas)

    # World model : entraîné si fourni, sinon aléatoire.
    if args.world_model and Path(args.world_model).is_file():
        params_wm = charger_world_model(args.world_model)
        journal.info("World model entraîné chargé : %s", args.world_model)
    elif args.champion and Path(args.champion).is_file():
        params_wm = charger_champion(args.champion)
        journal.info("Champion chargé comme world model : %s", args.champion)
    else:
        params_wm = initialiser_world_model(jax.random.PRNGKey(0))
        journal.warning("Aucun modèle fourni — world model ALÉATOIRE")

    planner = TDMPC2Planner(params_wm, nb_trajectoires=512, nb_iterations=2)
    sanitizer = ActionSanitizer()
    disjoncteur = DrawdownDisconnector(
        seuil_pct=4.0, capital_reference=EQUITY_INITIAL,
        chemin_journal="logs/endtoend_disjoncteur.jsonl",
    )

    equity = EQUITY_INITIAL
    position = 0.0  # lots signés
    prix_entree = 0.0
    nb_ordres = 0
    nb_refus = 0
    nb_disjonctions = 0
    trades_gagnants = 0
    nb_trades = 0
    moyenne_cem = None

    journal.info("Rejeu : barres [%d:%d] (%d pas)", debut, fin, fin - debut)
    for t in range(debut, fin):
        rapport = disjoncteur.verifier()
        if rapport.declenche:
            nb_disjonctions += 1
            if position != 0.0:
                pnl = position * CONTRAT * (prix[t] - prix_entree)
                equity += pnl
                position = 0.0
            disjoncteur.rearmement_manuel()
            continue

        latent = bridge_pytorch_to_jax(
            __import__("torch").from_numpy(latents[t]).contiguous()
        )
        cle = jax.random.PRNGKey(t)
        action, moyenne_cem = planner.planifier(cle, latent, moyenne_init=moyenne_cem)
        sig = np.asarray(action, dtype=np.float64)

        ordre = sanitizer.sanitiser(
            signal=sig, equity=equity, prix=float(prix[t]), distance_sl=DISTANCE_SL
        )
        if ordre.direction == 0:
            if "refusé" in ordre.raison:
                nb_refus += 1
            continue

        # Fermeture position opposée éventuelle.
        nouvelle_position = ordre.direction * ordre.lot
        if position != 0.0 and np.sign(nouvelle_position) != np.sign(position):
            pnl = position * CONTRAT * (prix[t] - prix_entree)
            equity += pnl
            nb_trades += 1
            if pnl > 0:
                trades_gagnants += 1
            disjoncteur.fermer_position(nb_trades, pnl)
            position = 0.0

        if position == 0.0 and ordre.lot >= sanitizer.limites.lot_min:
            position = nouvelle_position
            prix_entree = float(prix[t])
            nb_ordres += 1

    # Clôture finale.
    if position != 0.0:
        pnl = position * CONTRAT * (prix[fin - 1] - prix_entree)
        equity += pnl
        nb_trades += 1
        if pnl > 0:
            trades_gagnants += 1

    pnl_pct = (equity - EQUITY_INITIAL) / EQUITY_INITIAL * 100.0
    win_rate = trades_gagnants / max(1, nb_trades) * 100.0
    metriques = {
        "pnl_total_pct": pnl_pct,
        "equity_finale": equity,
        "nb_ordres": nb_ordres,
        "nb_trades": nb_trades,
        "nb_refus": nb_refus,
        "nb_disjonctions": nb_disjonctions,
        "win_rate": win_rate,
    }
    journal.info(
        "END-TO-END : pnl=%+.2f%% | equity=%.0f | ordres=%d trades=%d wr=%.1f%% "
        "refus=%d disjonctions=%d",
        pnl_pct, equity, nb_ordres, nb_trades, win_rate, nb_refus, nb_disjonctions,
    )
    return metriques


def main() -> None:
    """Point d'entrée CLI."""
    backtester(parse_args())


if __name__ == "__main__":
    main()
