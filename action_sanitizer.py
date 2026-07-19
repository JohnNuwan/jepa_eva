"""Couche d'action déterministe et disjoncteur dur — Bloc C d'E.V.A.

Deux garde-fous indépendants de l'IA, prêts pour la production :
- ``ActionSanitizer`` : intercepte les signaux continus issus du moteur JAX
  et applique un filtre déterministe sur la taille des lots selon une règle
  stricte de Risk Management (marge risquée ≤ 1 % de l'equity par trade) ;
- ``DrawdownDisconnector`` : disjoncteur dur qui surveille la perte
  journalière (latente + réalisée) et coupe TOUTES les positions puis
  suspend l'envoi d'ordres si le drawdown atteint 4 %.

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

journal = logging.getLogger("eva.sanitizer")

RISQUE_MAX_PCT: float = 1.0
SEUIL_DD_PCT: float = 4.0
PENALITE_FITNESS: float = -1_000.0


@dataclass(frozen=True)
class LimitesRisque:
    """Règles strictes de Risk Management par trade.

    Attributes:
        risque_max_pct: % maximal de l'equity risqué par trade (1 % spec).
        levier_max: Levier maximal autorisé (FTMO métaux/forex 1:100).
        lot_min: Taille minimale d'ordre.
        lot_max: Taille maximale d'ordre.
        taille_contrat: Unités sous-jacentes par lot standard (100 oz pour
            les métaux type XAUUSD ; 100 000 pour le forex).
    """

    risque_max_pct: float = RISQUE_MAX_PCT
    levier_max: float = 100.0
    lot_min: float = 0.01
    lot_max: float = 100.0
    taille_contrat: float = 100.0


@dataclass(frozen=True)
class OrdreValide:
    """Ordre final après filtrage déterministe du risque.

    Attributes:
        direction: -1 vendre, 0 neutre, +1 acheter.
        lot: Taille validée (écrasée si nécessaire).
        stop_loss: Prix du stop (0 si neutre).
        take_profit: Prix de l'objectif (0 si neutre).
        conforme: ``True`` si le signal d'origine respectait déjà les règles.
        raison: Justification lisible de l'écrasement éventuel.
    """

    direction: int
    lot: float
    stop_loss: float
    take_profit: float
    conforme: bool
    raison: str


class ActionSanitizer:
    """Filtre déterministe entre la sortie JAX et l'envoi de l'ordre.

    Traduit le signal continu de l'agent (direction, fraction de risque) en
    ordre borné par les règles de Risk Management. Toute taille de lot dont
    la perte au stop dépasserait ``risque_max_pct`` de l'equity est écrasée
    au lot de sécurité calculé ; l'écrasement est journalisé.
    """

    def __init__(self, limites: LimitesRisque | None = None) -> None:
        """Initialise le filtre avec les règles de risque.

        Args:
            limites: Règles de Risk Management ; défauts spec si ``None``.
        """
        self.limites = limites or LimitesRisque()

    def _lot_max_autorise(
        self, equity: float, prix: float, distance_sl: float
    ) -> float:
        """Calcule le lot maximal tel que la perte au SL ≤ risque autorisé.

        Deux bornes sont appliquées : le risque au stop
        (``risque_max_pct`` × equity) et le plafond de marge
        (equity × levier / notionnel).

        Args:
            equity: Capital du compte.
            prix: Prix actuel de l'actif.
            distance_sl: Distance absolue au stop loss (en points de prix).

        Returns:
            Taille de lot maximale autorisée par le risque, arrondie à 0.01
            (peut être inférieure à ``lot_min`` si le stop est très serré —
            le respect du risque prime sur la taille minimale).
        """
        risque_montant = equity * (self.limites.risque_max_pct / 100.0)
        lot_risque = risque_montant / (
            distance_sl * self.limites.taille_contrat
        )
        lot_marge = (equity * self.limites.levier_max) / (
            prix * self.limites.taille_contrat
        )
        lot = min(lot_risque, lot_marge)
        # Plafond absolu lot_max ; PAS de plancher ici : le respect du
        # risque prime sur lot_min (sinon un stop serré dépasserait 1 %).
        return float(min(np.floor(lot * 100.0) / 100.0, self.limites.lot_max))

    def sanitiser(
        self,
        signal: np.ndarray | list[float],
        equity: float,
        prix: float,
        distance_sl: float,
        ratio_tp: float = 2.0,
    ) -> OrdreValide:
        """Convertit le signal continu en ordre conforme au risque.

        ``signal[0]`` pilote la direction (tanh), ``signal[1]`` la fraction
        de risque souhaitée ∈ [0, 1]. Si le lot résultant dépasse le maximum
        autorisé, il est écrasé et l'écart est consigné dans ``raison``.

        Args:
            signal: Vecteur d'action continu issu du moteur JAX.
            equity: Capital du compte.
            prix: Prix actuel de l'actif.
            distance_sl: Distance absolue au stop loss.
            ratio_tp: Ratio TP/SL (défaut 2.0).

        Returns:
            ``OrdreValide`` prêt à être transmis au courtier.

        Raises:
            ValueError: Si ``equity``, ``prix`` ou ``distance_sl`` ≤ 0.
        """
        if equity <= 0:
            raise ValueError(f"equity > 0 requis, reçu {equity}")
        if prix <= 0:
            raise ValueError(f"prix > 0 requis, reçu {prix}")
        if distance_sl <= 0:
            raise ValueError(f"distance_sl > 0 requise, reçu {distance_sl}")

        signal_arr = np.asarray(signal, dtype=np.float64).ravel()
        direction_brute = (
            float(np.tanh(signal_arr[0])) if signal_arr.size else 0.0
        )
        # Fraction de risque : |signal[1]| bornée à [0, 1] — le CEM produit
        # des actions dans [-1, 1], la magnitude porte la conviction.
        fraction = (
            float(np.clip(abs(signal_arr[1]), 0.0, 1.0))
            if signal_arr.size > 1
            else 0.5
        )

        if abs(direction_brute) < 0.1:
            return OrdreValide(0, 0.0, 0.0, 0.0, True, "signal neutre")

        direction = 1 if direction_brute > 0 else -1
        # Lot souhaité borné par le risque autorisé (jamais par lot_max brut).
        lot_souhaite = self._lot_max_autorise(equity, prix, distance_sl) * fraction
        lot_max = self._lot_max_autorise(equity, prix, distance_sl)

        if lot_max < self.limites.lot_min:
            # Risque trop élevé même au lot minimal : ordre refusé.
            raison = (
                f"ordre refusé : lot max {lot_max:.2f} < lot_min "
                f"{self.limites.lot_min} (stop trop serré pour 1 %)"
            )
            journal.warning("ActionSanitizer : %s", raison)
            return OrdreValide(0, 0.0, 0.0, 0.0, False, raison)

        conforme = lot_souhaite <= lot_max
        lot_final = lot_souhaite if conforme else lot_max
        raison = (
            "conforme"
            if conforme
            else (
                f"lot écrasé {lot_souhaite:.2f} -> {lot_final:.2f} "
                f"(risque > {self.limites.risque_max_pct}%)"
            )
        )
        if not conforme:
            journal.warning("ActionSanitizer : %s", raison)

        if direction > 0:
            sl, tp = prix - distance_sl, prix + distance_sl * ratio_tp
        else:
            sl, tp = prix + distance_sl, prix - distance_sl * ratio_tp

        return OrdreValide(
            direction=direction,
            lot=float(lot_final),
            stop_loss=round(sl, 5),
            take_profit=round(tp, 5),
            conforme=conforme,
            raison=raison,
        )


@dataclass
class Position:
    """Position ouverte suivie par le disjoncteur.

    Attributes:
        ticket: Identifiant unique de la position.
        symbole: Symbole tradé.
        volume: Taille en lots.
        prix_ouverture: Prix d'entrée.
        profit_latent: Profit/perte non réalisé courant.
    """

    ticket: int
    symbole: str
    volume: float
    prix_ouverture: float
    profit_latent: float = 0.0


@dataclass
class RapportDisjonction:
    """Résultat d'une vérification du disjoncteur.

    Attributes:
        declenche: ``True`` si le seuil de drawdown est atteint.
        perte_pct: Perte journalière cumulée en % du capital de référence.
        positions_fermees: Tickets fermés lors de la disjonction.
        message: Description lisible de l'événement.
    """

    declenche: bool
    perte_pct: float
    positions_fermees: list[int] = field(default_factory=list)
    message: str = ""


class DrawdownDisconnector:
    """Disjoncteur dur : coupe tout si le drawdown quotidien ≥ seuil.

    Garde-fou absolu, non soumis à l'IA : à chaque vérification, si la perte
    latente + réalisée du jour atteint ``seuil_pct`` du capital de référence,
    toutes les positions sont fermées, l'envoi de nouveaux ordres est
    suspendu jusqu'à réarmement manuel, et l'événement est journalisé en
    JSONL avec une pénalité de fitness maximale pour la lignée fautive.
    """

    def __init__(
        self,
        seuil_pct: float = SEUIL_DD_PCT,
        capital_reference: float = 100_000.0,
        chemin_journal: str | Path = "logs/disjoncteur.jsonl",
    ) -> None:
        """Initialise le disjoncteur.

        Args:
            seuil_pct: Seuil de déclenchement en % du capital (4 % spec).
            capital_reference: Capital de référence du calcul de perte.
            chemin_journal: Fichier JSONL des événements de disjonction.
        """
        self.seuil_pct = seuil_pct
        self.capital_reference = capital_reference
        self.chemin_journal = Path(chemin_journal)
        self._suspendu = False
        self._positions: dict[int, Position] = {}
        self._perte_realisee_jour = 0.0
        self._jour = time.strftime("%Y-%m-%d")

    def _rollover_jour(self) -> None:
        """Réinitialise les compteurs au changement de jour de trading."""
        jour = time.strftime("%Y-%m-%d")
        if jour != self._jour:
            journal.info("Disjoncteur : rollover journalier %s -> %s", self._jour, jour)
            self._jour = jour
            self._perte_realisee_jour = 0.0
            self._suspendu = False

    def enregistrer_position(self, position: Position) -> None:
        """Ajoute une position ouverte à la surveillance.

        Args:
            position: Position à suivre.
        """
        self._positions[position.ticket] = position

    def maj_profit_latent(self, ticket: int, profit_latent: float) -> None:
        """Met à jour le profit latent d'une position suivie.

        Args:
            ticket: Ticket de la position.
            profit_latent: Profit/perte non réalisé courant.
        """
        if ticket in self._positions:
            self._positions[ticket].profit_latent = profit_latent

    def fermer_position(self, ticket: int, profit_realise: float) -> None:
        """Retire une position et accumule la perte réalisée du jour.

        Args:
            ticket: Ticket de la position fermée.
            profit_realise: P&L réalisé (négatif = perte).
        """
        self._positions.pop(ticket, None)
        if profit_realise < 0.0:
            self._perte_realisee_jour += abs(profit_realise)

    def perte_totale_pct(self) -> float:
        """Calcule la perte journalière totale en % du capital.

        Returns:
            (perte latente + perte réalisée) / capital_reference × 100.
        """
        perte_latente = sum(
            abs(p.profit_latent)
            for p in self._positions.values()
            if p.profit_latent < 0.0
        )
        return (
            (perte_latente + self._perte_realisee_jour)
            / self.capital_reference
            * 100.0
        )

    def verifier(self) -> RapportDisjonction:
        """Vérifie le seuil et déclenche la coupure totale si nécessaire.

        En cas de déclenchement : fermeture de toutes les positions
        suivies, suspension de l'envoi d'ordres, journalisation JSONL et
        pénalité de fitness maximale.

        Returns:
            ``RapportDisjonction`` détaillant l'état du garde-fou.
        """
        self._rollover_jour()
        perte_pct = self.perte_totale_pct()
        if perte_pct < self.seuil_pct and not self._suspendu:
            return RapportDisjonction(False, perte_pct, [], "sous le seuil")

        tickets = list(self._positions.keys())
        for ticket in tickets:
            pos = self._positions.pop(ticket)
            if pos.profit_latent < 0.0:
                self._perte_realisee_jour += abs(pos.profit_latent)
        self._suspendu = True
        message = (
            f"DISJONCTION : perte {perte_pct:.2f}% >= seuil "
            f"{self.seuil_pct}% — {len(tickets)} positions coupées, "
            f"ordres suspendus."
        )
        journal.critical("%s", message)
        self._journaliser(message, perte_pct, tickets)
        return RapportDisjonction(True, perte_pct, tickets, message)

    def autoriser_ordre(self) -> bool:
        """Indique si l'envoi d'un nouvel ordre est autorisé.

        Returns:
            ``True`` seulement si le disjoncteur n'a pas sauté aujourd'hui.
        """
        self._rollover_jour()
        return not self._suspendu

    def rearmement_manuel(self) -> None:
        """Réarme le disjoncteur (opération humaine, hors portée de l'IA)."""
        journal.warning("Disjoncteur : réarmement manuel")
        self._suspendu = False

    @staticmethod
    def penalite_fitness() -> float:
        """Retourne la pénalité de fitness maximale pour la lignée fautive.

        Returns:
            Pénalité constante appliquée à l'agent responsable.
        """
        return PENALITE_FITNESS

    def _journaliser(
        self, message: str, perte_pct: float, tickets: list[int]
    ) -> None:
        """Écrit l'événement de disjonction en JSONL.

        Args:
            message: Description de l'événement.
            perte_pct: Perte mesurée au déclenchement.
            tickets: Tickets des positions fermées.
        """
        entree = {
            "horodatage": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "message": message,
            "perte_pct": round(perte_pct, 4),
            "positions_fermees": tickets,
            "penalite_fitness": PENALITE_FITNESS,
        }
        try:
            self.chemin_journal.parent.mkdir(parents=True, exist_ok=True)
            with self.chemin_journal.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entree, ensure_ascii=False) + "\n")
        except OSError as exc:
            journal.error("Journal disjoncteur inaccessible : %s", exc)
