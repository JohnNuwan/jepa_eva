"""Chargeur de données historiques MT5 — flux OHLCV réel vers le pipeline.

Lit les CSV MT5 (time,open,high,low,close,tick_volume,spread,real_volume)
présents dans ``data/`` et produit des fenêtres glissantes normalisables par
``JEPAPipeline`` au format ``(Batch, Sequence, 5)`` = OHLCV.

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

COLONNES_OHLCV: tuple[str, ...] = ("open", "high", "low", "close", "tick_volume")


def charger_csv(chemin: str | Path) -> dict[str, np.ndarray]:
    """Charge un CSV MT5 en dictionnaire de tableaux float32.

    Args:
        chemin: Chemin vers le CSV (ex. ``data/XAUUSD_m15.csv``).

    Returns:
        Dictionnaire ``{colonne: ndarray float32}`` trié par temps croissant,
        incluant au minimum les 5 colonnes OHLCV.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas.
        ValueError: Si une colonne OHLCV est absente.
    """
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(f"CSV introuvable : {chemin}")
    donnees = np.genfromtxt(
        chemin, delimiter=",", names=True, dtype=None, encoding="utf-8"
    )
    noms = donnees.dtype.names or ()
    manquantes = [c for c in COLONNES_OHLCV if c not in noms]
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans {chemin.name} : {manquantes}")
    sortie: dict[str, np.ndarray] = {}
    for nom in noms:
        col = donnees[nom]
        if col.dtype.kind in ("f", "i", "u"):
            sortie[nom] = col.astype(np.float32)
    return sortie


class FenetreurHistorique:
    """Produit des fenêtres glissantes OHLCV ``(B, T, 5)`` depuis un CSV.

    Attributes:
        ohlcv: Matrice complète ``(nb_barres, 5)`` float32.
        longueur: Longueur de chaque fenêtre.
        nb_barres: Nombre total de barres chargées.
    """

    def __init__(self, chemin: str | Path, longueur: int = 128) -> None:
        """Charge le CSV et prépare le fenêtrage.

        Args:
            chemin: Chemin vers le CSV MT5.
            longueur: Longueur de fenêtre (128 barres par défaut).

        Raises:
            ValueError: Si le CSV contient moins de ``longueur`` barres.
        """
        donnees = charger_csv(chemin)
        self.ohlcv = np.stack(
            [donnees[c] for c in COLONNES_OHLCV], axis=-1
        ).astype(np.float32)
        self.nb_barres = int(self.ohlcv.shape[0])
        if self.nb_barres < longueur:
            raise ValueError(
                f"{self.nb_barres} barres < longueur fenêtre {longueur}"
            )
        self.longueur = longueur

    def fenetre(self, debut: int) -> Tensor:
        """Extrait une fenêtre ``(1, longueur, 5)`` démarrant à ``debut``.

        Args:
            debut: Indice de la première barre de la fenêtre.

        Returns:
            Tenseur ``(1, longueur, 5)`` float32.

        Raises:
            IndexError: Si la fenêtre dépasse les données disponibles.
        """
        fin = debut + self.longueur
        if debut < 0 or fin > self.nb_barres:
            raise IndexError(
                f"fenêtre [{debut}:{fin}] hors limites [0:{self.nb_barres}]"
            )
        return torch.from_numpy(self.ohlcv[debut:fin]).unsqueeze(0)

    def lots(
        self, taille_batch: int, debut: int = 0, pas: int = 1
    ) -> Tensor:
        """Empile ``taille_batch`` fenêtres espacées de ``pas`` barres.

        Args:
            taille_batch: Nombre de fenêtres du batch.
            debut: Indice de départ de la première fenêtre.
            pas: Décalage entre fenêtres consécutives.

        Returns:
            Tenseur ``(taille_batch, longueur, 5)`` float32.

        Raises:
            IndexError: Si le batch dépasse les données.
        """
        fenetres = [
            self.ohlcv[debut + i * pas : debut + i * pas + self.longueur]
            for i in range(taille_batch)
        ]
        empile = np.stack(fenetres, axis=0)
        if empile.shape[1] != self.longueur:
            raise IndexError(
                f"batch [{debut}:{debut + taille_batch * pas}] hors limites"
            )
        return torch.from_numpy(empile)
