#!/usr/bin/env python3
"""
Connecteur MT5 Bridge — remplace le stub de connecteur_mt5.py dans JEPA_EVA.
Parle au MT5 Bridge sur 192.168.1.6:8765 (le PC local).

Interface identique au stub existant pour compatibilité avec main.py.
"""
from __future__ import annotations

import logging
import time
import json
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("eva.broker.mt5")

MT5_BRIDGE = "http://192.168.1.6:8765"

@dataclass
class TickBroker:
    symbole: str
    bid: float
    ask: float
    volume: float
    horodatage: float = field(default_factory=time.time)

@dataclass
class ResultatOrdre:
    succes: bool
    ticket: int = 0
    prix_execution: float = 0.0
    message: str = ""

class ConnecteurMT5:
    """Connecteur MT5 réel via le Bridge local."""

    def __init__(self, bridge_url: str = MT5_BRIDGE):
        self.bridge_url = bridge_url
        self._cache_positions: list[dict] = []
        self._cache_time = 0.0
        self._cache_ttl = 1.0  # 1 seconde de cache

    def _requete(self, endpoint: str, data: Optional[dict] = None) -> dict:
        """Envoie une requête HTTP au MT5 Bridge."""
        url = f"{self.bridge_url}/{endpoint}"
        try:
            if data:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers={"Content-Type": "application/json"},
                )
            else:
                req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            logger.warning("MT5 Bridge %s error: %s", endpoint, e)
            return {}

    def positions(self) -> list[dict]:
        """Récupère les positions ouvertes (avec cache court)."""
        now = time.time()
        if now - self._cache_time < self._cache_ttl:
            return self._cache_positions
        result = self._requete("positions")
        self._cache_positions = result.get("positions", [])
        self._cache_time = now
        return self._cache_positions

    def compte(self) -> dict:
        """Infos du compte MT5."""
        return self._requete("account")

    def modifier_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> ResultatOrdre:
        """Modifie SL/TP d'une position."""
        result = self._requete("modify", {"ticket": ticket, "sl": sl, "tp": tp})
        if result.get("success"):
            return ResultatOrdre(succes=True, ticket=ticket, message="Modifié OK")
        return ResultatOrdre(succes=False, message=result.get("error", "Erreur inconnue"))

    def fermer_position(self, ticket: int, volume: Optional[float] = None) -> ResultatOrdre:
        """Ferme une position."""
        data = {"ticket": ticket}
        if volume is not None:
            data["volume"] = volume
        result = self._requete("close", data)
        if result.get("success"):
            return ResultatOrdre(succes=True, ticket=ticket, message="Fermé OK")
        return ResultatOrdre(succes=False, message=result.get("error", "Erreur inconnue"))

    def envoyer_ordre(self, symbole: str, direction: int, lot: float, sl: float, tp: float) -> ResultatOrdre:
        """
        Envoie un ordre directement au MT5 Bridge (192.168.1.6:8765).
        Le bridge exécute l'ordre sur MT5/FTMO.
        """
        from urllib.request import Request, urlopen
        from urllib.error import URLError
        url = f"{self.bridge_url}/trade"
        payload = json.dumps({
            "symbol": symbole,
            "type": "BUY" if direction > 0 else "SELL",
            "volume": lot,
            "sl": sl,
            "tp": tp,
            "magic": 234567,
            "comment": "JEPA-EVA",
        }).encode()
        try:
            req = Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                if result.get("retcode") == 10009 or result.get("success"):
                    logger.info("ORDRE EXECUTE: %s %s %.2f lots SL=%.2f TP=%.2f", symbole, "BUY" if direction > 0 else "SELL", lot, sl, tp)
                    return ResultatOrdre(succes=True, ticket=result.get("ticket", 0), message="Ordre exécuté")
                else:
                    logger.warning("ORDRE REJETE: %s", result.get("error", "inconnu"))
                    return ResultatOrdre(succes=False, message=result.get("error", "Rejeté"))
        except URLError as e:
            logger.error("Erreur MT5 Bridge: %s", e)
            return ResultatOrdre(succes=False, message=str(e))

    def tick(self, symbole: str) -> Optional[TickBroker]:
        """Récupère le dernier tick réel depuis le MT5 Bridge."""
        try:
            from urllib.request import urlopen
            url = f"{self.bridge_url}/tick/{symbole}"
            with urlopen(url, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                if "error" in data:
                    logger.warning("Tick erreur: %s", data["error"])
                    return None
                return TickBroker(
                    symbole=data.get("symbol", symbole),
                    bid=float(data.get("bid", 0.0)),
                    ask=float(data.get("ask", 0.0)),
                    volume=1.0,
                )
        except Exception as e:
            logger.warning("Impossible de récupérer le tick: %s", e)
            return None


# === Pour compatibilité avec le stub existant ===

@dataclass
class _PosSim:
    symbole: str
    direction: int
    lot: float
    prix_entree: float
    sl: float
    tp: float


class ConnecteurStub:
    """Stub MT5 (simulation) — remplacé par ConnecteurMT5 pour le live."""

    def __init__(self):
        self._positions: list[_PosSim] = []
        self._compteur = 0
        logger.info("ConnecteurStub initialisé (mode simulation)")

    def positions(self) -> list[dict]:
        return [
            {"ticket": s.prix_entree, "symbol": s.symbole, "type": "BUY" if s.direction == 0 else "SELL",
             "volume": s.lot, "open_price": s.prix_entree, "current_price": s.prix_entree,
             "sl": s.sl, "tp": s.tp, "profit": 0.0, "swap": 0.0}
            for s in self._positions
        ]

    def envoyer_ordre(self, symbole: str, direction: int, lot: float, sl: float, tp: float) -> ResultatOrdre:
        self._compteur += 1
        self._positions.append(_PosSim(symbole, direction, lot, 100.0, sl, tp))
        logger.info("[STUB] Ordre simulé: %s %s %s lots", symbole, direction, lot)
        return ResultatOrdre(succes=True, ticket=self._compteur, prix_execution=100.0)


# === Factory ===
def creer_connecteur(mode: str = "auto") -> ConnecteurMT5 | ConnecteurStub:
    """Crée le connecteur approprié. 'auto' = MT5 si le bridge répond, sinon stub."""
    if mode == "reel":
        return ConnecteurMT5()
    if mode == "stub":
        return ConnecteurStub()

    # Auto-détection
    try:
        c = ConnecteurMT5()
        compte = c.compte()
        if compte.get("login"):
            logger.info("MT5 Bridge détecté - mode réel activé")
            return c
    except Exception:
        pass
    logger.warning("MT5 Bridge non joignable - fallback mode simulation")
    return ConnecteurStub()