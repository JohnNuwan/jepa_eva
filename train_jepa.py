"""Pré-entraînement long de l'encodeur JEPA sur données réelles MT5.

Entraîne le ``TimeJEPAEncoder`` (via ``MomentumTarget``) en auto-supervisé
sur les fenêtres OHLCV historiques : perte Smooth L1 sur bloc futur masqué
20 %, mise à jour EMA de la cible, AdamW + cosine schedule, gradient
clipping, checkpoints périodiques et sauvegarde de l'encodeur figé.

Usage :
    PYTHONPATH=. venv/bin/python train_jepa.py --symbole XAUUSD --steps 2000

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

from donnees_reelles import FenetreurHistorique
from jepa_pipeline import JEPAPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.train_jepa")

DOSSIER_CHECKPOINTS = Path("checkpoints_jepa")


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de la ligne de commande.

    Returns:
        Espace de noms avec symbole, timeframe, hyperparamètres et chemins.
    """
    p = argparse.ArgumentParser(description="Pré-entraînement JEPA MT5")
    p.add_argument("--symbole", default="XAUUSD")
    p.add_argument("--timeframe", default="m15")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--longueur", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sauvegarde", type=int, default=200, help="période checkpoint")
    p.add_argument("--sortie", default=str(DOSSIER_CHECKPOINTS))
    return p.parse_args()


def planificateur_cosine(step: int, total: int, lr_base: float, warmup: int = 50) -> float:
    """Calcule le taux d'apprentissage (warmup linéaire + décroissance cosine).

    Args:
        step: Itération courante.
        total: Nombre total d'itérations.
        lr_base: Taux de base après warmup.
        warmup: Durée du warmup linéaire.

    Returns:
        Taux d'apprentissage à appliquer.
    """
    if step < warmup:
        return lr_base * step / max(1, warmup)
    progression = (step - warmup) / max(1, total - warmup)
    return lr_base * 0.5 * (1.0 + np.cos(np.pi * min(1.0, progression)))


def entrainer(args: argparse.Namespace) -> dict[str, list[float]]:
    """Boucle principale de pré-entraînement JEPA.

    Args:
        args: Arguments de la ligne de commande.

    Returns:
        Historique ``{"pertes": [...], "lrs": [...]}`` pour la courbe.

    Raises:
        FileNotFoundError: Si le CSV du symbole est absent.
    """
    chemin_csv = Path("data") / f"{args.symbole}_{args.timeframe}.csv"
    if not chemin_csv.is_file():
        raise FileNotFoundError(f"CSV introuvable : {chemin_csv}")

    fen = FenetreurHistorique(chemin_csv, longueur=args.longueur)
    pipeline = JEPAPipeline(device=args.device)
    params = [p for p in pipeline.modele.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-5)

    dossier = Path(args.sortie)
    dossier.mkdir(parents=True, exist_ok=True)
    historique: dict[str, list[float]] = {"pertes": [], "lrs": []}
    nb_max_debut = fen.nb_barres - args.longueur - args.batch

    journal.info(
        "Début : %s %s | %d barres | %d steps | batch=%d | lr=%.1e",
        args.symbole, args.timeframe, fen.nb_barres, args.steps, args.batch, args.lr,
    )
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()

    for step in range(args.steps):
        lr = planificateur_cosine(step, args.steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        debut = int(rng.integers(0, nb_max_debut))
        batch = fen.lots(args.batch, debut=debut, pas=1).to(args.device)

        perte = pipeline.pas_entrainement(batch)
        opt.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        historique["pertes"].append(float(perte))
        historique["lrs"].append(lr)

        if (step + 1) % 50 == 0:
            moy = float(np.mean(historique["pertes"][-50:]))
            journal.info(
                "step %4d/%d | perte(50)=%.5f | lr=%.2e | %.0f step/s",
                step + 1, args.steps, moy, lr,
                (step + 1) / max(1e-9, time.perf_counter() - t0),
            )
        if (step + 1) % args.sauvegarde == 0:
            ckpt = dossier / f"jepa_step{step + 1}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "encodeur": pipeline.modele.encodeur_online.state_dict(),
                    "normalisateur": pipeline.normalisateur.state_dict(),
                    "perte": float(perte),
                },
                ckpt,
            )
            journal.info("checkpoint -> %s", ckpt)

    # Sauvegarde finale : encodeur figé pour l'inférence (pont DLPack).
    final = dossier / f"jepa_final_{args.symbole}_{args.timeframe}.pt"
    torch.save(
        {
            "encodeur": pipeline.modele.encodeur_online.state_dict(),
            "normalisateur": pipeline.normalisateur.state_dict(),
            "symbole": args.symbole,
            "timeframe": args.timeframe,
            "steps": args.steps,
            "perte_finale": historique["pertes"][-1],
        },
        final,
    )
    with (dossier / "historique_pertes.json").open("w", encoding="utf-8") as f:
        json.dump(historique, f)

    moy_debut = float(np.mean(historique["pertes"][:50]))
    moy_fin = float(np.mean(historique["pertes"][-50:]))
    journal.info(
        "Terminé : perte %.5f -> %.5f | encodeur -> %s", moy_debut, moy_fin, final
    )
    return historique


def main() -> None:
    """Point d'entrée CLI."""
    entrainer(parse_args())


if __name__ == "__main__":
    main()
