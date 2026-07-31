"""Connecteur broker MT5/cTrader — interface unique + implémentation stub.

Définit le contrat entre l'orchestrateur E.V.A et le terminal de trading
(MetaTrader 5 via VM Windows KVM, ou cTrader en fallback). L'implémentation
``ConnecteurStub`` simule le broker en mémoire pour les tests end-to-end
sans infrastructure réelle ; ``ConnecteurMT5`` (ZMQ/EA) sera branché sur la
VM Windows quand elle sera disponible.

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

journal = logging.getLogger("eva.broker")


@dataclass
class _PosSim:
    """Position simulée interne du stub."""

    symbole: str
    direction: int
    lot: float
    prix_entree: float
    sl: float
    tp: float


@dataclass
class TickBroker:
    """Un tick de marché reçu du broker.

    Attributes:
        symbole: Symbole tradé.
        bid: Prix vendeur.
        ask: Prix acheteur.
        volume: Volume du tick.
        horodatage: Timestamp Unix du tick.
    """

    symbole: str
    bid: float
    ask: float
    volume: float
    horodatage: float = field(default_factory=time.time)


@dataclass
class ResultatOrdre:
    """Résultat d'un envoi d'ordre au broker.

    Attributes:
        succes: ``True`` si l'ordre est accepté/exécuté.
        ticket: Identifiant de la position ouverte (0 si échec).
        prix_execution: Prix réel d'exécution.
        message: Description (raison d'échec éventuelle).
    """

    succes: bool
    ticket: int
    prix_execution: float
    message: str


class ConnecteurBroker(ABC):
    """Interface abstraite broker (MT5, cTrader, ou stub de test)."""

    @abstractmethod
    def connecter(self) -> bool:
        """Établit la connexion au terminal. Retourne ``True`` si OK."""

    @abstractmethod
    def deconnecter(self) -> None:
        """Ferme proprement la connexion."""

    @abstractmethod
    def tick_courant(self, symbole: str) -> TickBroker | None:
        """Retourne le dernier tick du symbole, ou ``None`` si indisponible."""

    @abstractmethod
    def envoyer_ordre(
        self,
        symbole: str,
        direction: int,
        lot: float,
        stop_loss: float,
        take_profit: float,
    ) -> ResultatOrdre:
        """Envoie un ordre de marché avec SL/TP.

        Args:
            symbole: Symbole à trader.
            direction: +1 achat, -1 vente.
            lot: Taille en lots.
            stop_loss: Prix du stop.
            take_profit: Prix de l'objectif.

        Returns:
            ``ResultatOrdre`` avec ticket et prix d'exécution.
        """

    @abstractmethod
    def fermer_position(self, ticket: int) -> bool:
        """Ferme une position par ticket. Retourne ``True`` si OK."""

    @abstractmethod
    def equity(self) -> float:
        """Retourne l'equity courante du compte."""

    @abstractmethod
    def positions_ouvertes(self) -> list[int]:
        """Retourne la liste des tickets de positions ouvertes."""


class ConnecteurStub(ConnecteurBroker):
    """Broker simulé en mémoire pour tests end-to-end sans infra réelle.

    Simule l'exécution d'ordres avec slippage léger et le suivi des
    positions. Remplace ``flux_marche_simule`` par un flux piloté depuis
    l'orchestrateur via ``injecter_prix``.
    """

    def __init__(self, equity_initiale: float = 100_000.0) -> None:
        """Initialise le stub.

        Args:
            equity_initiale: Capital de départ du compte simulé.
        """
        self._equity = equity_initiale
        self._prix: dict[str, float] = {}
        self._positions: dict[int, _PosSim] = {}
        self._prochain_ticket = 1
        self._connecte = False

    def connecter(self) -> bool:
        """Marque le stub comme connecté.

        Returns:
            Toujours ``True``.
        """
        self._connecte = True
        journal.info("ConnecteurStub : connecté (broker simulé)")
        return True

    def deconnecter(self) -> None:
        """Déconnecte le stub."""
        self._connecte = False
        journal.info("ConnecteurStub : déconnecté")

    def injecter_prix(self, symbole: str, bid: float, ask: float | None = None) -> None:
        """Injecte un prix (alimente le flux simulé).

        Args:
            symbole: Symbole concerné.
            bid: Prix vendeur.
            ask: Prix acheteur (défaut bid + spread minimal).
        """
        self._prix[symbole] = bid
        self._prix[f"{symbole}_ask"] = ask if ask is not None else bid * 1.0001

    def tick_courant(self, symbole: str) -> TickBroker | None:
        """Retourne le tick simulé du symbole.

        Args:
            symbole: Symbole demandé.

        Returns:
            ``TickBroker`` ou ``None`` si prix non injecté.
        """
        if symbole not in self._prix:
            return None
        return TickBroker(
            symbole=symbole,
            bid=self._prix[symbole],
            ask=self._prix.get(f"{symbole}_ask", self._prix[symbole]),
            volume=100.0,
        )

    def envoyer_ordre(
        self,
        symbole: str,
        direction: int,
        lot: float,
        stop_loss: float,
        take_profit: float,
    ) -> ResultatOrdre:
        """Simule l'exécution d'un ordre avec slippage léger.

        Args:
            symbole: Symbole à trader.
            direction: +1 achat, -1 vente.
            lot: Taille en lots.
            stop_loss: Prix du stop.
            take_profit: Prix de l'objectif.

        Returns:
            ``ResultatOrdre`` avec ticket attribué.
        """
        if not self._connecte:
            return ResultatOrdre(False, 0, 0.0, "non connecté")
        tick = self.tick_courant(symbole)
        if tick is None:
            return ResultatOrdre(False, 0, 0.0, f"prix {symbole} indisponible")
        # Slippage : exécution à ask (achat) ou bid (vente).
        prix_exec = tick.ask if direction > 0 else tick.bid
        ticket = self._prochain_ticket
        self._prochain_ticket += 1
        self._positions[ticket] = _PosSim(
            symbole=symbole,
            direction=direction,
            lot=lot,
            prix_entree=prix_exec,
            sl=stop_loss,
            tp=take_profit,
        )
        journal.info(
            "Stub ORDRE #%d : %s %+d %.2f @ %.2f (SL=%.2f TP=%.2f)",
            ticket, symbole, direction, lot, prix_exec, stop_loss, take_profit,
        )
        return ResultatOrdre(True, ticket, prix_exec, "exécuté (stub)")

    def fermer_position(self, ticket: int) -> bool:
        """Ferme une position simulée et réalise le P&L.

        Args:
            ticket: Ticket de la position.

        Returns:
            ``True`` si la position existait et est fermée.
        """
        pos = self._positions.pop(ticket, None)
        if pos is None:
            return False
        tick = self.tick_courant(pos.symbole)
        if tick is not None:
            prix_sortie = tick.bid if pos.direction > 0 else tick.ask
            pnl = (
                pos.direction
                * pos.lot
                * 100.0  # contrat XAUUSD
                * (prix_sortie - pos.prix_entree)
            )
            self._equity += pnl
            journal.info("Stub CLOSE #%d : pnl=%+.2f", ticket, pnl)
        return True

    def equity(self) -> float:
        """Retourne l'equity simulée.

        Returns:
            Equity courante.
        """
        return self._equity

    def positions_ouvertes(self) -> list[int]:
        """Retourne les tickets des positions simulées ouvertes.

        Returns:
            Liste de tickets.
        """
        return list(self._positions.keys())


class ConnecteurMT5(ConnecteurBroker):
    """Connecteur MetaTrader 5 via EA ZMQ (VM Windows KVM) — à implémenter.

    Nécessite la VM Windows avec MT5 + un Expert Advisor ZMQ/JSON. Les
    méthodes lèvent ``NotImplementedError`` tant que la VM n'est pas prête.
    """

    def __init__(self, hote: str = "192.168.122.1", port: int = 5555) -> None:
        """Initialise l'adresse du terminal MT5.

        Args:
            hote: IP de la VM Windows (réseau KVM par défaut).
            port: Port ZMQ de l'EA MT5.
        """
        self.hote = hote
        self.port = port

    def connecter(self) -> bool:
        """Non implémenté — requiert la VM Windows MT5.

        Raises:
            NotImplementedError: Toujours, tant que la VM n'est pas prête.
        """
        raise NotImplementedError(
            "ConnecteurMT5 : VM Windows MT5 requise (ISO dans ~/vms/mt5/iso/)"
        )

    def deconnecter(self) -> None:
        """Non implémenté."""
        raise NotImplementedError("VM MT5 requise")

    def tick_courant(self, symbole: str) -> TickBroker | None:
        """Non implémenté."""
        raise NotImplementedError("VM MT5 requise")

    def envoyer_ordre(
        self, symbole: str, direction: int, lot: float,
        stop_loss: float, take_profit: float,
    ) -> ResultatOrdre:
        """Non implémenté."""
        raise NotImplementedError("VM MT5 requise")

    def fermer_position(self, ticket: int) -> bool:
        """Non implémenté."""
        raise NotImplementedError("VM MT5 requise")

    def equity(self) -> float:
        """Non implémenté."""
        raise NotImplementedError("VM MT5 requise")

    def positions_ouvertes(self) -> list[int]:
        """Non implémenté."""
        raise NotImplementedError("VM MT5 requise")
