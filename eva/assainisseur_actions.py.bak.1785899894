"""ActionSanitizer — interface déterministe entre la sortie JAX et l'ordre.

Analyse le signal continu généré par le planificateur et écrase toute taille
de lot qui violerait les règles strictes de Money Management (risque de
marge > 1 % du compte par trade), forçant un lot de sécurité.

Conforme au Bloc C.1 de la MASTER-SPECIFICATION E.V.A.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LimitesMoneyManagement:
    """Règles de risque par trade."""

    risque_max_pct: float = 1.0  # % du compte risqué par trade
    levier_max: float = 100.0  # levier maximal autorisé
    lot_min: float = 0.01
    lot_max: float = 100.0
    tick_valeur: float = 1.0  # valeur d'un point en devise du compte
    taille_contrat: float = 100_000.0  # taille d'un lot standard


@dataclass(frozen=True)
class DecisionSanitisee:
    """Ordre final après validation des règles de risque."""

    direction: int  # -1 vendre, 0 neutre, +1 acheter
    lot: float  # taille de lot validée
    stop_loss: float  # prix SL (0 si neutre)
    take_profit: float  # prix TP (0 si neutre)
    raison: str  # justification de l'écrasement éventuel


class ActionSanitizer:
    """Traduit le signal continu de l'agent en ordre conforme au risque.

    Args:
        limites: règles de Money Management à appliquer.
    """

    def __init__(self, limites: LimitesMoneyManagement | None = None) -> None:
        """Initialise l'assainisseur avec les règles de risque.

        Args:
            limites: Règles de Money Management ; valeurs par défaut si
                ``None``.
        """
        self.limites = limites or LimitesMoneyManagement()

    def _calculer_lot_max(
        self,
        equity: float,
        prix: float,
        distance_sl: float,
    ) -> float:
        """Lot maximal tel que la perte au SL ≤ risque_max_pct × equity.

        Args:
            equity: capital du compte.
            prix: prix actuel de l'actif.
            distance_sl: distance absolue au stop loss.

        Returns:
            Taille de lot maximale autorisée (arrondie à 0.01).
        """
        risque_montant = equity * (self.limites.risque_max_pct / 100.0)
        if distance_sl <= 0.0:
            return self.limites.lot_min
        # Perte au SL = lot × taille_contrat × distance_sl (en devise compte).
        lot = risque_montant / (distance_sl * self.limites.taille_contrat)
        # Plafond de marge : notionnel ≤ equity × levier.
        notionnel_max = equity * self.limites.levier_max
        lot_marge = notionnel_max / (prix * self.limites.taille_contrat)
        lot = min(lot, lot_marge)
        lot = float(np.floor(lot * 100.0) / 100.0)
        return float(np.clip(lot, self.limites.lot_min, self.limites.lot_max))

    def sanitiser(
        self,
        signal: np.ndarray | list[float],
        equity: float,
        prix: float,
        distance_sl: float,
        ratio_tp: float = 2.0,
    ) -> DecisionSanitisee:
        """Convertit le signal continu de l'agent en ordre validé.

        La première composante du signal donne la direction (tanh), la
        seconde la fraction de risque souhaitée. Si le lot résultant dépasse
        le maximum autorisé par le Money Management, il est écrasé et la
        raison est consignée.

        Args:
            signal: Vecteur d'action continu, ``signal[0]`` = direction,
                ``signal[1]`` = fraction de risque ∈ [0, 1].
            equity: Capital du compte en devise de référence.
            prix: Prix actuel de l'actif.
            distance_sl: Distance absolue au stop loss (en points de prix).
            ratio_tp: Ratio TP/SL (défaut 2.0).

        Returns:
            ``DecisionSanitisee`` avec lot écrasé si nécessaire.

        Raises:
            ValueError: Si ``equity``, ``prix`` ou ``distance_sl`` sont
                négatifs ou nuls.
        """
        if equity <= 0:
            raise ValueError(f"equity > 0 requis, reçu {equity}")
        if prix <= 0:
            raise ValueError(f"prix > 0 requis, reçu {prix}")
        if distance_sl <= 0:
            raise ValueError(f"distance_sl > 0 requise, reçu {distance_sl}")
        signal_arr = np.asarray(signal, dtype=np.float64).ravel()
        direction_brute = float(np.tanh(signal_arr[0])) if signal_arr.size else 0.0
        fraction_risque = float(np.clip(signal_arr[1], 0.0, 1.0)) if signal_arr.size > 1 else 0.5

        if abs(direction_brute) < 0.1:
            return DecisionSanitisee(
                direction=0,
                lot=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                raison="signal neutre",
            )

        direction = 1 if direction_brute > 0 else -1
        lot_souhaite = self.limites.lot_max * fraction_risque
        lot_max_autorise = self._calculer_lot_max(equity, prix, distance_sl)

        raison = "conforme"
        lot_final = lot_souhaite
        if lot_souhaite > lot_max_autorise:
            lot_final = lot_max_autorise
            raison = (
                f"lot écrasé {lot_souhaite:.2f} -> {lot_final:.2f} "
                f"(risque > {self.limites.risque_max_pct}%)"
            )
        lot_final = float(np.clip(lot_final, self.limites.lot_min, self.limites.lot_max))

        if direction > 0:
            sl = prix - distance_sl
            tp = prix + distance_sl * ratio_tp
        else:
            sl = prix + distance_sl
            tp = prix - distance_sl * ratio_tp

        return DecisionSanitisee(
            direction=direction,
            lot=lot_final,
            stop_loss=round(sl, 5),
            take_profit=round(tp, 5),
            raison=raison,
        )
