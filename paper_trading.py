#!/usr/bin/env python3
"""
Paper Trading E.V.A — Multi-symbol live test sans ordres réels
BTCUSDT gen99 / SOLUSDT gen9 / XAUUSD gen4

Usage :
    cd ~/jepa_eva && PYTHONPATH=. venv/bin/python paper_trading.py

Boucle : chaque tick M15 → charge les données récentes → encode JEPA →
    planifie CEM → sanitizer → log ordre virtuel → track P&L
"""

import json
import logging
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import OrderedDict

import numpy as np
import jax
import jax.numpy as jnp
import requests

from jax_arena import (
    TDMPC2Planner,
    bridge_pytorch_to_jax,
    initialiser_world_model,
    ParametresWorldModel,
)
from action_sanitizer import ActionSanitizer
from backtest_validation import charger_champion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("eva.paper")

# === CONFIG ===
CAPITAL_INITIAL = 100_000.0  # $ virtuel
JEPA_CHECKPOINT = "checkpoints_jepa/jepa_final_{symbole}_m15.pt"
JEPA_SYMBOLES = {"BTCUSDT": "BTCUSDT", "SOLUSDT": "SOLUSDT", "HYPUSDT": "HYPUSDT", "XAUUSD": "XAUUSD"}
CHAMPIONS = {
    "BTCUSDT": "registry_arena_validated/champions/champion_gen99.npz",
    "SOLUSDT": "registry_arena_validated/champions/champion_gen9.npz",
    "HYPUSDT": "registry_arena_validated/champions/champion_gen4.npz",
    "XAUUSD": "registry_arena_validated/champions/champion_gen4.npz",
}
ALLOCATION = {"BTCUSDT": 0.35, "SOLUSDT": 0.20, "HYPUSDT": 0.20, "XAUUSD": 0.25}
FENETRE = 128
LOGS_PATH = Path("logs/paper_trading.jsonl")


class DataLiveBinance:
    """Récupère les M15 récents depuis Binance REST (crypto) + Yahoo/Bybit."""

    def __init__(self):
        self.url = "https://api.binance.com/api/v3/klines"

    def rafraichir(self) -> dict[str, np.ndarray]:
        """Retourne {symbole: ohlcv_array (N,5)}."""
        result = {}
        # BTC + SOL depuis Binance
        for s in ["BTCUSDT", "SOLUSDT"]:
            try:
                resp = requests.get(self.url, params={"symbol": s, "interval": "15m", "limit": FENETRE + 10}, timeout=10)
                data = resp.json()
                if data and isinstance(data, list):
                    arr = np.array([[float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data], dtype=np.float32)
                    result[s] = arr
            except Exception:
                pass

        # HYP depuis Bybit
        try:
            resp = requests.get("https://api.bybit.com/v5/market/kline",
                                params={"category": "spot", "symbol": "HYPEUSDT", "interval": "15", "limit": FENETRE + 10}, timeout=10)
            data = resp.json()
            if data.get("retCode") == 0:
                batch = list(reversed(data["result"]["list"]))
                arr = np.array([[float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in batch], dtype=np.float32)
                result["HYPUSDT"] = arr
        except Exception:
            pass

        # XAU via Yahoo GLD ×10
        try:
            import yfinance as yf
            gld = yf.download("GLD", period="2d", interval="15m")
            if not gld.empty:
                o = gld["Open"].values.astype(np.float32) * 10
                h = gld["High"].values.astype(np.float32) * 10
                l_ = gld["Low"].values.astype(np.float32) * 10
                c = gld["Close"].values.astype(np.float32) * 10
                v = gld["Volume"].values.astype(np.float32)
                result["XAUUSD"] = np.column_stack([o, h, l_, c, v])
        except Exception:
            pass

        return result


class DataLiveYahoo:
    """Récupère les M15 de GC=F (or) depuis Yahoo Finance."""

    def __init__(self):
        self.cache = []

    def rafraichir(self) -> np.ndarray:
        try:
            import yfinance as yf
            xau = yf.download("GC=F", period="2d", interval="15m")
            if xau.empty:
                return np.array([])
            # Format: OHLCV numpy
            o = xau["Open"].values.astype(np.float32)
            h = xau["High"].values.astype(np.float32)
            l_ = xau["Low"].values.astype(np.float32)
            c = xau["Close"].values.astype(np.float32)
            v = xau["Volume"].values.astype(np.float32)
            return np.column_stack([o, h, l_, c, v])
        except Exception as e:
            log.warning("Yahoo XAU: %s", e)
            return np.array([])


class PortefeuillePaper:
    """Suit les positions virtuelles et le P&L."""

    def __init__(self, capital: float, allocation: dict):
        self.capital = capital
        self.allocation = allocation
        self.positions: dict[str, dict] = {}  # symbole -> {"direction": ±1, "prix_entree": float, "lots": float}
        self.trades = []
        self.equity = capital
        self.tick = 0

    def executer_ordre(self, symbole: str, direction: int, lots: float, prix: float):
        if direction == 0 or lots <= 0:
            return
        pos = self.positions.get(symbole)
        if pos and np.sign(pos["direction"]) != np.sign(direction):
            # Fermeture position opposée
            pnl = pos["direction"] * pos["lots"] * (prix - pos["prix_entree"])
            self.equity += pnl
            self.trades.append({"symbole": symbole, "pnl": pnl, "prix_sortie": prix})
            log.info("📊 %s FERMÉ: pnl=%+.2f$ equity=%.0f$", symbole, pnl, self.equity)
            self.positions.pop(symbole)

        if direction != 0 and symbole not in self.positions:
            capital_alloue = self.equity * self.allocation.get(symbole, 0.2)
            lots_reels = min(lots, capital_alloue / prix * 0.1)  # ~10x leverage max
            self.positions[symbole] = {"direction": direction, "prix_entree": prix, "lots": lots_reels}
            log.info("📈 %s OUVERT: dir=%+d lots=%.4f à %.2f$", symbole, direction, lots_reels, prix)

    def mark_to_market(self, prix: dict[str, float]):
        """Met à jour l'equity avec les prix courants."""
        unrealized = 0.0
        for s, pos in self.positions.items():
            p = prix.get(s)
            if p:
                unrealized += pos["direction"] * pos["lots"] * (p - pos["prix_entree"])
        return self.equity + unrealized

    def rapport(self) -> str:
        lines = [f"\n{'='*50}", "PORTEFEUILLE PAPER", f"{'='*50}"]
        lines.append(f"Capital initial: {CAPITAL_INITIAL:.0f}$")
        lines.append(f"Equity: {self.equity:.0f}$")
        lines.append(f"P&L: {self.equity - CAPITAL_INITIAL:+.0f}$ ({(self.equity - CAPITAL_INITIAL) / CAPITAL_INITIAL * 100:+.2f}%)")
        lines.append(f"Trades: {len(self.trades)}")
        if self.trades:
            wins = sum(1 for t in self.trades if t["pnl"] > 0)
            lines.append(f"Win rate: {wins / len(self.trades) * 100:.1f}%")
        lines.append(f"Positions ouvertes: {len(self.positions)}")
        for s, pos in self.positions.items():
            lines.append(f"  {s}: dir={pos['direction']:+d} lots={pos['lots']:.4f} entrée={pos['prix_entree']:.2f}")
        return "\n".join(lines)


def main():
    log.info("🚀 Paper Trading E.V.A — Multi-symbol Live")
    log.info("Capital: %.0f$ | Symboles: BTCUSDT, SOLUSDT", CAPITAL_INITIAL)

    # Data feeds
    binance = DataLiveBinance()
    portefeuille = PortefeuillePaper(CAPITAL_INITIAL, ALLOCATION)

    # Charger les champions
    champions: dict[str, ParametresWorldModel] = {}
    for s, chemin in CHAMPIONS.items():
        if Path(chemin).is_file():
            champions[s] = charger_champion(chemin)
            log.info("✅ %s champion chargé: %s", s, chemin)
        else:
            log.warning("⚠️ %s champion absent: %s", s, chemin)

    # Initialiser les planificateurs
    planners: dict[str, TDMPC2Planner] = {}
    for s, params in champions.items():
        planners[s] = TDMPC2Planner(params, nb_trajectoires=512, nb_iterations=2)
    sanitizer = ActionSanitizer()

    t0 = time.time()
    derniere_barre: dict[str, str] = {}
    buffers: dict[str, list] = {s: [] for s in CHAMPIONS}

    # Charger le pipeline JEPA une fois
    log.info("🧠 Chargement des pipelines JEPA...")
    from jepa_pipeline import JEPAPipeline
    pipelines: dict[str, JEPAPipeline] = {}
    for s in CHAMPIONS:
        try:
            ckpt = JEPA_CHECKPOINT.format(symbole=JEPA_SYMBOLES.get(s, s))
            if not Path(ckpt).is_file():
                log.warning("⚠️ %s checkpoint absent: %s", s, ckpt)
                continue
            pipe = JEPAPipeline(device="cuda:0")
            ckpt_data = __import__("torch").load(ckpt, map_location="cuda:0", weights_only=False)
            pipe.modele.encodeur_online.load_state_dict(ckpt_data["encodeur"])
            pipe.normalisateur.load_state_dict(ckpt_data["normalisateur"])
            pipe.modele.eval()
            pipe.normalisateur.eval()
            pipelines[s] = pipe
            log.info("✅ %s JEPA pipeline chargé (perte=%.5f)", s, ckpt_data.get("perte_finale", 0))
        except Exception as e:
            log.warning("⚠️ %s JEPA pipeline: %s", s, e)

    try:
        while True:
            # Récupérer les données live
            ohlcv_dict = binance.rafraichir()
            if not ohlcv_dict:
                log.info("⏳ Attente données...")
                time.sleep(30)
                continue

            prix_actuels = {}
            nouvelles_barres = False

            for s, ohlcv in ohlcv_dict.items():
                if s not in pipelines or s not in planners:
                    continue
                if len(ohlcv) < FENETRE:
                    continue

                # Buffer glissant de 128 barres
                buffers[s] = ohlcv[-FENETRE:].tolist()
                if len(buffers[s]) < FENETRE:
                    continue

                nouvelles_barres = True
                prix_courant = float(ohlcv[-1, 3])
                prix_actuels[s] = prix_courant

                # Encodage JEPA réel
                import torch
                batch = torch.from_numpy(np.array(buffers[s], dtype=np.float32)).unsqueeze(0).to("cuda:0")
                with torch.no_grad():
                    try:
                        latents = pipelines[s].encoder(batch)  # (1, 128, 128)
                        latent = latents[0, -1, :].contiguous().cpu().numpy()
                    except Exception as e:
                        log.warning("⚠️ %s encode: %s", s, e)
                        continue

                # Planifier avec le champion
                try:
                    latent_jax = bridge_pytorch_to_jax(
                        torch.from_numpy(latent).contiguous()
                    )
                    cle = jax.random.PRNGKey(portefeuille.tick)
                    action, _ = planners[s].planifier(cle, latent_jax)
                except Exception as e:
                    log.warning("⚠️ %s planif: %s", s, e)
                    continue

                # Sanitizer
                signal = np.asarray(action, dtype=np.float64)
                ordre = sanitizer.sanitiser(
                    signal=signal, equity=portefeuille.equity,
                    prix=prix_courant, distance_sl=5.0,
                )

                if ordre.direction != 0 and ordre.lot > 0:
                    portefeuille.executer_ordre(s, ordre.direction, float(ordre.lot), prix_courant)

            if nouvelles_barres:
                equity_mtm = portefeuille.mark_to_market(prix_actuels)
                log.info("🔄 Tick %d | Equity MTM: %.0f$ | P&L: %+.2f%% | Positions: %d",
                         portefeuille.tick, equity_mtm,
                         (equity_mtm - CAPITAL_INITIAL) / CAPITAL_INITIAL * 100,
                         len(portefeuille.positions))

            portefeuille.tick += 1
            time.sleep(60)  # Vérifier toutes les minutes

    except KeyboardInterrupt:
        log.info("🛑 Arrêt demandé")
        log.info(portefeuille.rapport())


if __name__ == "__main__":
    main()