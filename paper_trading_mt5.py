#!/usr/bin/env python3
"""
Paper Trading MT5 Bridge — Test des 20 meilleurs champions JAX sur le marché réel.

Chaque champion reçoit 100 000 $ virtuels, trade sur les données MT5 réelles
via le bridge MT5 (http://192.168.1.6:8765), et génère un classement continu.

Usage:
    cd /home/aza/projects/jepa_eva
    PYTHONPATH=. python paper_trading_mt5.py                   # CPU (défaut)
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python paper_trading_mt5.py --gpu  # GPU 0

Boucle M15 :
    OHLCV → JEPA encode → CEM planifier → Sanitizer → P&L virtuel → Ranking
"""

import json
import logging
import os
import sys
import time
import math
import urllib.request
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import jax

# ── JAX device ──────────────────────────────────────────────────────────────
USE_GPU = "--gpu" in sys.argv
if not USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    jax.config.update("jax_platform_name", "cpu")
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.12"

DEVICE_STR = "cuda:0" if USE_GPU else "cpu"

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("eva.paper.mt5")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
PAPER_DIR = DATA_DIR / "paper_results"
CHAMPIONS_DIR = DATA_DIR / "paper_champions"
SELECTION_FILE = DATA_DIR / "champions_selection.json"
RANKING_FILE = PAPER_DIR / "rankings.json"

PAPER_DIR.mkdir(parents=True, exist_ok=True)

# ── Bridge MT5 ──────────────────────────────────────────────────────────────
BRIDGE_URL = "http://192.168.1.6:8765"
TIMEOUT = 15

# ── Per-symbol point value (dollars per 1.0 lot per 1.0 price unit) ────────
# P&L = lot × Δprice × POINT_VALUE[symbol]
POINT_VALUES = {
    "EURUSD": 100000,
    "GBPUSD": 100000,
    "USDJPY": 100000,  # adjusted dynamically by rate
    "XAUUSD": 100,     # 1 lot = 100 oz, $1 move = $100/lot
    "US30.cash": 1,    # 1 point = $1 per lot
    "US100.cash": 1,
    "US500.cash": 1,
    "GER40.cash": 1,
}

# ── ATR periods for SL distance ─────────────────────────────────────────────
ATR_PERIOD = 14
SL_ATR_MULT = 1.5
TP_SL_RATIO = 2.0
MIN_SL_DIST = {
    "EURUSD": 0.0010,
    "GBPUSD": 0.0010,
    "USDJPY": 0.10,
    "XAUUSD": 1.0,
    "US30.cash": 10.0,
    "US100.cash": 10.0,
    "US500.cash": 5.0,
    "GER40.cash": 10.0,
}

# ── Virtual capital ─────────────────────────────────────────────────────────
CAPITAL_INITIAL = 100_000.0
FENETRE = 128


# ═══════════════════════════════════════════════════════════════════════════════
# Data sources
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(symbol: str, bars: int = FENETRE + 10, tf: str = "M15") -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fetch OHLCV bars from MT5 bridge.

    Returns:
        (ohlcv, times) where ohlcv is (N, 5) with [open, high, low, close, volume]
        and times is (N,) int64 unix timestamps.
    """
    url = f"{BRIDGE_URL}/ohlcv/{symbol}/{bars}/{tf}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode())
        if "bars" in data and data["bars"]:
            times = np.array([b["time"] for b in data["bars"]], dtype=np.int64)
            arr = np.array([[b["open"], b["high"], b["low"], b["close"], b["volume"]]
                           for b in data["bars"]], dtype=np.float32)
            return arr, times
        return None, None
    except Exception as e:
        log.warning("OHLCV fetch error %s: %s", symbol, e)
        return None, None


def compute_atr(bars: np.ndarray, period: int = ATR_PERIOD) -> float:
    """Compute ATR from OHLCV bars."""
    if len(bars) < period + 1:
        return 0.0
    high, low, close = bars[-period - 1:, 1], bars[-period - 1:, 2], bars[-period - 1:, 3]
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    return float(np.mean(tr))


def get_point_value(symbol: str, current_price: float = None) -> float:
    """Get point value for a symbol, handling dynamic USDJPY case."""
    pv = POINT_VALUES.get(symbol, 100000)
    if symbol == "USDJPY" and current_price and current_price > 0:
        # 1 lot = 100000 JPY, price in JPY per USD
        # Δprice = 1 USDJPY = 100000 JPY = 100000/rate USD
        pv = 100000.0 / current_price
    return pv


# ═══════════════════════════════════════════════════════════════════════════════
# Virtual Account
# ═══════════════════════════════════════════════════════════════════════════════

class ComptePaper:
    """Virtual trading account for one champion."""

    def __init__(self, champion_id: str, symbol: str, capital: float = CAPITAL_INITIAL):
        self.champion_id = champion_id
        self.symbol = symbol
        self.capital = capital
        self.equity = capital
        self.positions: list[dict] = []  # open positions
        self.trades: list[dict] = []     # closed trades
        self.tick = 0
        self.peak_equity = capital
        self.max_drawdown = 0.0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.wins = 0
        self.losses = 0

    def open_position(self, direction: int, lots: float, price: float,
                      sl: float, tp: float, point_value: float):
        """Open a virtual position."""
        if direction == 0 or lots <= 0:
            return
        self.positions.append({
            "ticket": self.tick,
            "symbol": self.symbol,
            "direction": direction,
            "lots": lots,
            "open_price": price,
            "sl": sl,
            "tp": tp,
            "point_value": point_value,
            "open_time": datetime.now().isoformat(),
        })

    def mark_to_market(self, current_price: float) -> float:
        """Update equity with unrealized P&L at current price."""
        unrealized = 0.0
        for pos in self.positions:
            pv = pos.get("point_value", get_point_value(self.symbol, current_price))
            move = current_price - pos["open_price"]
            pnl = pos["direction"] * pos["lots"] * move * pv
            pos["unrealized_pnl"] = pnl
            unrealized += pnl
        self.equity = self.capital + unrealized + sum(t["pnl"] for t in self.trades)
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, dd)
        return self.equity

    def check_sl_tp(self, current_price: float) -> list[dict]:
        """Check and close positions that hit SL/TP."""
        closed = []
        remaining = []
        for pos in self.positions:
            pv = pos.get("point_value", get_point_value(self.symbol, current_price))
            if pos["direction"] > 0:  # BUY
                hit_sl = pos["sl"] > 0 and current_price <= pos["sl"]
                hit_tp = pos["tp"] > 0 and current_price >= pos["tp"]
            else:  # SELL
                hit_sl = pos["sl"] > 0 and current_price >= pos["sl"]
                hit_tp = pos["tp"] > 0 and current_price <= pos["tp"]

            if hit_sl or hit_tp:
                move = current_price - pos["open_price"]
                pnl = pos["direction"] * pos["lots"] * move * pv
                reason = "SL" if hit_sl else "TP"
                closed.append({
                    **pos,
                    "close_price": current_price,
                    "pnl": pnl,
                    "close_time": datetime.now().isoformat(),
                    "close_reason": reason,
                })
                self.trades.append(closed[-1])
                if pnl > 0:
                    self.wins += 1
                    self.total_profit += pnl
                else:
                    self.losses += 1
                    self.total_loss += abs(pnl)
            else:
                remaining.append(pos)

        self.positions = remaining
        return closed

    def close_all(self, current_price: float) -> list[dict]:
        """Close all open positions (e.g., for end-of-test)."""
        closed = []
        for pos in self.positions:
            pv = pos.get("point_value", get_point_value(self.symbol, current_price))
            move = current_price - pos["open_price"]
            pnl = pos["direction"] * pos["lots"] * move * pv
            closed.append({
                **pos,
                "close_price": current_price,
                "pnl": pnl,
                "close_time": datetime.now().isoformat(),
                "close_reason": "FORCE_CLOSE",
            })
            self.trades.append(closed[-1])
        self.positions = []
        self.equity = self.capital + sum(t["pnl"] for t in self.trades)
        return closed

    def get_metrics(self) -> dict:
        """Compute performance metrics for ranking."""
        total_pnl = sum(t["pnl"] for t in self.trades)
        total_trades = len(self.trades)
        unrealized = sum(p.get("unrealized_pnl", 0) for p in self.positions)
        total_equity = self.capital + total_pnl + unrealized
        total_return = (total_equity - self.capital) / self.capital * 100

        # Sharpe ratio (daily approximation)
        if total_trades >= 2:
            pnl_series = np.array([t["pnl"] for t in self.trades[-50:]])
            if np.std(pnl_series) > 0:
                sharpe = np.mean(pnl_series) / np.std(pnl_series) * np.sqrt(96)  # 96 M15 bars per day
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0.0
        profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else float('inf')

        return {
            "champion_id": self.champion_id,
            "symbol": self.symbol,
            "capital": round(self.capital, 2),
            "equity": round(total_equity, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return, 2),
            "total_trades": total_trades,
            "open_positions": len(self.positions),
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else None,
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(self.max_drawdown, 2),
            "peak_equity": round(self.peak_equity, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Main paper trading engine
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("Paper Trading MT5 Bridge — Test des champions JAX")
    log.info(f"Device: {'GPU' if USE_GPU else 'CPU'}")
    log.info(f"Capital virtuel: {CAPITAL_INITIAL:,.0f}$ par champion")
    log.info("=" * 60)

    # ── Load selection ───────────────────────────────────────────────────────
    if not SELECTION_FILE.is_file():
        log.error("Selection file not found: %s", SELECTION_FILE)
        sys.exit(1)

    with open(SELECTION_FILE) as f:
        selection = json.load(f)

    top20 = selection["top20"]
    log.info("Champions chargés: %d", len(top20))
    for i, e in enumerate(top20):
        log.info("  %2d. %s | %s | fitness=%.2f", i + 1,
                 e["symbol"], e["run_id"], e["fitness"])

    # ── Group champions by symbol (shared encoder) ───────────────────────────
    sym_to_champs: dict[str, list[dict]] = {}
    for e in top20:
        s = e["symbol"]
        sym_to_champs.setdefault(s, []).append(e)

    symbols = list(sym_to_champs.keys())
    log.info("Symboles: %s", symbols)

    # ── Load JEPA pipelines (one per symbol, shared) ─────────────────────────
    log.info("Chargement des pipelines JEPA...")
    import torch

    # Check which checkpoints exist
    chkpt_dir = BASE / "checkpoints_jepa"
    pipelines: dict[str, any] = {}
    for s in symbols:
        ckpt = chkpt_dir / f"jepa_final_{s}_m15.pt"
        if not ckpt.is_file():
            log.warning("  ⚠️ %s: checkpoint absent: %s", s, ckpt)
            continue
        try:
            from jepa_pipeline import JEPAPipeline
            pipe = JEPAPipeline(device=DEVICE_STR)
            ckpt_data = torch.load(ckpt, map_location=DEVICE_STR, weights_only=False)
            pipe.modele.encodeur_online.load_state_dict(ckpt_data["encodeur"])
            pipe.normalisateur.load_state_dict(ckpt_data["normalisateur"])
            pipe.modele.eval()
            pipe.normalisateur.eval()
            pipelines[s] = pipe
            log.info("  ✅ %s: JEPA pipeline chargé (perte=%.5f)",
                     s, ckpt_data.get("perte_finale", 0))
        except Exception as ex:
            log.warning("  ⚠️ %s: erreur chargement JEPA: %s", s, ex)

    if not pipelines:
        log.error("AUCUN pipeline JEPA chargé. Abandon.")
        sys.exit(1)

    # ── Load champion params and create planners ─────────────────────────────
    log.info("Chargement des champions et création des planificateurs CEM...")
    from jax_arena import (
        TDMPC2Planner, bridge_pytorch_to_jax, ParametresWorldModel
    )
    from backtest_validation import charger_champion

    champions: list[dict] = []
    for e in top20:
        cp = e["champion_path"]
        if not os.path.isfile(cp):
            # Try alternative path
            alt = CHAMPIONS_DIR / os.path.basename(cp)
            if alt.is_file():
                cp = str(alt)
            else:
                log.warning("  ⚠️ %s: champion absent: %s", e["run_id"], cp)
                continue

        if e["symbol"] not in pipelines:
            log.warning("  ⚠️ %s: pas de pipeline pour %s", e["run_id"], e["symbol"])
            continue

        try:
            params = charger_champion(cp)
            planner = TDMPC2Planner(params, nb_trajectoires=512, nb_iterations=2)
            champion = {
                "entry": e,
                "params": params,
                "planner": planner,
                "account": ComptePaper(e["run_id"], e["symbol"]),
            }
            champions.append(champion)
            log.info("  ✅ %s: champion chargé (fitness=%.2f, %s)",
                     e["run_id"], e["fitness"], e["symbol"])
        except Exception as ex:
            log.warning("  ⚠️ %s: erreur chargement: %s", e["run_id"], ex)

    if not champions:
        log.error("AUCUN champion chargé. Abandon.")
        sys.exit(1)

    log.info("Champions actifs: %d / %d", len(champions), len(top20))

    # ── Import sanitizer ─────────────────────────────────────────────────────
    from action_sanitizer import ActionSanitizer, LimitesRisque

    # ── Buffers ──────────────────────────────────────────────────────────────
    buffers: dict[str, list] = {s: [] for s in symbols}
    derniere_barre_time: dict[str, int] = {}
    dernier_rapport = time.time()

    # ── Main loop ────────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("Démarrage de la boucle de paper trading M15")
    log.info("─" * 60)

    try:
        while True:
            tick_time = time.time()
            nouvelles_barres = False

            for s in symbols:
                # ── Fetch OHLCV ──────────────────────────────────────────────
                ohlcv = fetch_ohlcv(s, FENETRE + 10, "M15")
                if ohlcv is None or len(ohlcv) < FENETRE:
                    continue

                # Detect new bar by last close time
                last_time = int(ohlcv[-1, 0]) if ohlcv.shape[1] > 4 else 0
                if s in derniere_barre_time and derniere_barre_time[s] == last_time:
                    continue  # no new bar yet

                nouvelles_barres = True
                derniere_barre_time[s] = last_time
                buffers[s] = ohlcv[-FENETRE:].tolist()

                # Current price (close of last bar)
                current_price = float(ohlcv[-1, 3])
                point_value = get_point_value(s, current_price)
                atr = compute_atr(ohlcv[-ATR_PERIOD - 2:])
                sl_dist = max(atr * SL_ATR_MULT, MIN_SL_DIST.get(s, atr * SL_ATR_MULT))

                # ── Encode JEPA (once per symbol) ────────────────────────────
                if s not in pipelines:
                    continue
                pipe = pipelines[s]
                try:
                    batch = torch.from_numpy(
                        np.array(buffers[s], dtype=np.float32)
                    ).unsqueeze(0).to(DEVICE_STR)
                    with torch.no_grad():
                        latents = pipe.encoder(batch)
                        latent = latents[0, -1, :].contiguous().cpu().numpy()
                except Exception as ex:
                    log.warning("  ⚠️ %s encode: %s", s, ex)
                    continue

                # ── Process each champion for this symbol ────────────────────
                for c in champions:
                    if c["entry"]["symbol"] != s:
                        continue
                    champ_id = c["entry"]["run_id"]
                    account = c["account"]
                    account.tick += 1

                    # Check SL/TP on existing positions
                    closed = account.check_sl_tp(current_price)
                    for cl in closed:
                        log.info("  📊 %s %s: %s pnl=%.2f$",
                                 champ_id, cl["close_reason"], s, cl["pnl"])

                    # Plan with CEM
                    try:
                        latent_torch = torch.from_numpy(latent).contiguous()
                        latent_jax = bridge_pytorch_to_jax(latent_torch)
                        cle = jax.random.PRNGKey(account.tick)
                        action, _ = c["planner"].planifier(cle, latent_jax)
                    except Exception as ex:
                        log.warning("  ⚠️ %s planif: %s", champ_id, ex)
                        continue

                    # Sanitize
                    sanitizer = ActionSanitizer(
                        LimitesRisque(
                            risque_max_pct=1.0,
                            levier_max=30.0,
                            lot_min=0.01,
                            lot_max=100.0,
                            taille_contrat=point_value,
                        )
                    )
                    signal = np.asarray(action, dtype=np.float64)
                    try:
                        ordre = sanitizer.sanitiser(
                            signal=signal,
                            equity=account.equity,
                            prix=current_price,
                            distance_sl=sl_dist,
                            ratio_tp=TP_SL_RATIO,
                        )
                    except Exception as ex:
                        log.warning("  ⚠️ %s sanitizer: %s", champ_id, ex)
                        continue

                    if ordre.direction != 0 and ordre.lot > 0:
                        account.open_position(
                            direction=ordre.direction,
                            lots=float(ordre.lot),
                            price=current_price,
                            sl=ordre.stop_loss,
                            tp=ordre.take_profit,
                            point_value=point_value,
                        )
                        log.info("  📈 %s OUVERT %s dir=%+d lots=%.2f sl=%.2f tp=%.2f",
                                 champ_id, s, ordre.direction, ordre.lot,
                                 ordre.stop_loss, ordre.take_profit)

                    # Mark to market
                    account.mark_to_market(current_price)

            # ── Periodic report (every 15 min) ───────────────────────────────
            if nouvelles_barres or (time.time() - dernier_rapport > 900):
                update_ranking(champions)
                dernier_rapport = time.time()

            if not nouvelles_barres:
                time.sleep(30)
            else:
                time.sleep(5)

    except KeyboardInterrupt:
        log.info("🛑 Arrêt demandé")

    # ── Final report ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("RAPPORT FINAL")
    log.info("=" * 60)
    for c in champions:
        m = c["account"].get_metrics()
        log.info("%s | %s | equity=%.0f$ | P&L=%+.2f$ | trades=%d | WR=%.1f%% | Sharpe=%.3f | DD=%.2f%%",
                 m["champion_id"], m["symbol"], m["equity"], m["total_pnl"],
                 m["total_trades"], m["win_rate"], m["sharpe_ratio"],
                 m["max_drawdown_pct"])

    update_ranking(champions, final=True)
    log.info("Classement final: %s", RANKING_FILE)


def update_ranking(champions: list, final: bool = False):
    """Update the continuous ranking file."""
    metrics = [c["account"].get_metrics() for c in champions]
    metrics.sort(key=lambda m: m["total_pnl"], reverse=True)

    ranking = {
        "timestamp": datetime.now().isoformat(),
        "final": final,
        "total_champions": len(metrics),
        "ranking": metrics,
    }

    try:
        with open(RANKING_FILE, "w") as f:
            json.dump(ranking, f, indent=2, default=str)
        log.info("Ranking updated: %d champions, leader=%s (P&L=%.2f$)",
                 len(metrics), metrics[0]["champion_id"] if metrics else "N/A",
                 metrics[0]["total_pnl"] if metrics else 0)

        # Also log per-champion journal
        for m in metrics:
            journal_file = PAPER_DIR / f"{m['champion_id']}.jsonl"
            entry = {k: v for k, v in m.items() if k != "champion_id"}
            entry["timestamp"] = datetime.now().isoformat()
            with open(journal_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    except Exception as ex:
        log.warning("Ranking update error: %s", ex)


if __name__ == "__main__":
    main()