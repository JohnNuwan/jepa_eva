"""DrawdownDisconnecter — disjoncteur ultime non soumis à l'IA.

Surveille la perte latente (non-réalisée) et réelle (réalisée) cumulée de la
journée. Si elle atteint le seuil critique (4 % par défaut), coupe
instantanément toutes les positions, suspend l'envoi de nouveaux ordres, et
enregistre une pénalité de fitness maximale pour la lignée fautive.

Conforme au Bloc C.2 de la MASTER-SPECIFICATION E.V.A.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PositionOuverte:
    """Position suivie par le disjoncteur."""

    ticket: int
    symbole: str
    volume: float
    prix_ouverture: float
    profit_latent: float = 0.0


@dataclass
class RapportDisjonction:
    """Résultat d'une vérification de sécurité."""

    declenche: bool
    perte_pct: float
    positions_fermees: list[int] = field(default_factory=list)
    message: str = ""


class DrawdownDisconnecter:
    """Garde-fou absolu : coupe tout si la perte journalière ≥ seuil.

    Args:
        seuil_pct: seuil de déclenchement en % du capital (4 % par spec).
        capital_reference: capital de référence pour le calcul de la perte.
        chemin_journal: fichier JSONL où enregistrer les disjonctions.
    """

    def __init__(
        self,
        seuil_pct: float = 4.0,
        capital_reference: float = 100_000.0,
        chemin_journal: str | Path = "disjoncteur_journal.jsonl",
    ) -> None:
        """Initialise le disjoncteur de drawdown.

        Args:
            seuil_pct: Seuil de déclenchement en % du capital (4 % par spec).
            capital_reference: Capital de référence pour le calcul de perte.
            chemin_journal: Fichier JSONL des disjonctions.
        """
        self.seuil_pct = seuil_pct
        self.capital_reference = capital_reference
        self.chemin_journal = Path(chemin_journal)
        self._suspendu = False
        self._positions: dict[int, PositionOuverte] = {}
        self._perte_realisee_jour = 0.0
        self._dernier_jour = time.strftime("%Y-%m-%d")

    def _reinitialiser_si_nouveau_jour(self) -> None:
        jour = time.strftime("%Y-%m-%d")
        if jour != self._dernier_jour:
            self._perte_realisee_jour = 0.0
            self._dernier_jour = jour
            self._suspendu = False

    def enregistrer_position(self, position: PositionOuverte) -> None:
        """Ajoute une position ouverte à la surveillance."""
        self._positions[position.ticket] = position

    def fermer_position(self, ticket: int, profit_realise: float) -> None:
        """Retire une position et accumule la perte réalisée."""
        self._positions.pop(ticket, None)
        if profit_realise < 0.0:
            self._perte_realisee_jour += abs(profit_realise)

    def maj_profit_latent(self, ticket: int, profit_latent: float) -> None:
        """Met à jour le profit latent d'une position suivie."""
        if ticket in self._positions:
            self._positions[ticket].profit_latent = profit_latent

    def perte_totale_pct(self) -> float:
        """Perte cumulée (latente + réalisée) en % du capital de référence."""
        perte_latente = sum(
            abs(p.profit_latent) for p in self._positions.values() if p.profit_latent < 0
        )
        return (perte_latente + self._perte_realisee_jour) / self.capital_reference * 100.0

    def verifier(self) -> RapportDisjonction:
        """Vérifie le seuil et déclenche la coupure si nécessaire.

        Returns:
            ``RapportDisjonction`` indiquant si le disjoncteur a sauté.
        """
        self._reinitialiser_si_nouveau_jour()
        perte_pct = self.perte_totale_pct()

        if perte_pct < self.seuil_pct and not self._suspendu:
            return RapportDisjonction(
                declenche=False,
                perte_pct=perte_pct,
                message="sous le seuil",
            )

        # Déclenchement : fermeture de toutes les positions.
        tickets_fermes = list(self._positions.keys())
        for ticket in tickets_fermes:
            pos = self._positions.pop(ticket)
            if pos.profit_latent < 0.0:
                self._perte_realisee_jour += abs(pos.profit_latent)

        self._suspendu = True
        message = (
            f"DISJONCTION : perte {perte_pct:.2f}% >= seuil {self.seuil_pct}%. "
            f"{len(tickets_fermes)} positions fermées."
        )
        self._journaliser(message, perte_pct, tickets_fermes)
        return RapportDisjonction(
            declenche=True,
            perte_pct=perte_pct,
            positions_fermees=tickets_fermes,
            message=message,
        )

    def autoriser_nouvel_ordre(self) -> bool:
        """Retourne ``True`` seulement si le disjoncteur n'a pas sauté."""
        self._reinitialiser_si_nouveau_jour()
        return not self._suspendu

    def rearmement_manuel(self) -> None:
        """Réarme le disjoncteur (opération manuelle hors IA)."""
        self._suspendu = False

    def penalite_fitness(self) -> float:
        """Pénalité maximale à attribuer à la lignée fautive."""
        return -1_000.0

    def _journaliser(
        self, message: str, perte_pct: float, tickets: list[int]
    ) -> None:
        entree = {
            "horodatage": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "message": message,
            "perte_pct": round(perte_pct, 4),
            "positions_fermees": tickets,
            "penalite_fitness": self.penalite_fitness(),
        }
        self.chemin_journal.parent.mkdir(parents=True, exist_ok=True)
        with self.chemin_journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entree, ensure_ascii=False) + "\n")
