"""Pré-calcul des latents JEPA de tout l'historique avec l'encodeur figé.

Encode l'intégralité des barres d'un symbole en une seule passe (encodeur
pré-entraîné chargé depuis ``checkpoints_jepa/``), et sauvegarde les latents
128-dim alignés avec les prix de clôture pour l'entraînement de l'arène.

Usage :
    PYTHONPATH=. venv/bin/python precompute_latents.py --symbole XAUUSD

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from donnees_reelles import charger_csv
from jepa_pipeline import JEPAPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.precompute")

COLONNES: tuple[str, ...] = ("open", "high", "low", "close", "tick_volume")


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI.

    Returns:
        Espace de noms avec symbole, timeframe, chemins et device.
    """
    p = argparse.ArgumentParser(description="Pré-calcul latents JEPA")
    p.add_argument("--symbole", default="XAUUSD")
    p.add_argument("--timeframe", default="m15")
    p.add_argument("--checkpoint", default="checkpoints_jepa/jepa_final_XAUUSD_m15.pt")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sortie", default="latents")
    return p.parse_args()


def precompute(args: argparse.Namespace) -> Path:
    """Encode tout l'historique et sauvegarde prix + latents alignés.

    Args:
        args: Arguments CLI.

    Returns:
        Chemin du fichier ``.npz`` produit.

    Raises:
        FileNotFoundError: Si le CSV ou le checkpoint est absent.
    """
    chemin_csv = Path("data") / f"{args.symbole}_{args.timeframe}.csv"
    if not chemin_csv.is_file():
        raise FileNotFoundError(f"CSV introuvable : {chemin_csv}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {args.checkpoint}")

    donnees = charger_csv(chemin_csv)
    nb = len(donnees["close"])
    pipeline = JEPAPipeline(device=args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    pipeline.modele.encodeur_online.load_state_dict(ckpt["encodeur"])
    pipeline.normalisateur.load_state_dict(ckpt["normalisateur"])
    pipeline.modele.eval()
    journal.info("Encodeur chargé (perte=%.5f)", ckpt.get("perte_finale", float("nan")))

    ohlcv = np.stack([donnees[c] for c in COLONNES], axis=-1).astype(np.float32)
    latents_liste: list[np.ndarray] = []
    # Chunk limité à la longueur du position embedding (512).
    taille_chunk = 512
    with torch.no_grad():
        for debut in range(0, nb, taille_chunk):
            chunk = torch.from_numpy(ohlcv[debut : debut + taille_chunk]).unsqueeze(0).to(args.device)
            lat = pipeline.encoder(chunk)[0]  # (chunk, 128)
            latents_liste.append(lat.cpu().numpy())
    latents = np.concatenate(latents_liste, axis=0).astype(np.float32)

    dossier = Path(args.sortie)
    dossier.mkdir(parents=True, exist_ok=True)
    sortie = dossier / f"{args.symbole}_{args.timeframe}_latents.npz"
    np.savez_compressed(
        sortie,
        prix=donnees["close"].astype(np.float32),
        latents=latents,
        symbole=args.symbole,
        timeframe=args.timeframe,
    )
    journal.info("Latents %s sauvegardés -> %s", latents.shape, sortie)
    return sortie


def main() -> None:
    """Point d'entrée CLI."""
    precompute(parse_args())


if __name__ == "__main__":
    main()
