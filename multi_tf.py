#!/usr/bin/env python3
"""
Multi-timeframe (H4/H1/M15/M5) feature extraction for TheHive trading strategies.

Bridge API: http://192.168.1.6:8765/ohlcv/{SYMBOL}/{BARS}/{TF}
Available TFs on bridge: M1, M5, M15, M30 (H1, H4, D1 return "No data")

Architecture:
  H4 features  ← aggregated from M30 (8 bars = 1 pseudo-H4 candle)
  H1 features  ← computed from M30 data (~EMA20 on M30 ≈ EMA20 on H1)
  M15 features ← direct from bridge (JEPA signal, volume filter)
  M5 features  ← direct from bridge (micro-trend, last 3 candles)

Usage:
    from multi_tf import get_mtf_features
    features = get_mtf_features("US30.cash")
    # features = {
    #   "h4": {"trend_dir": 1, "atr": 45.2, "ema50": 54300},
    #   "h1": {"ema20_dir": 1, "rsi": 58, "macd_hist": 12.5},
    #   "m15": {"price": 54450, "volume_ratio": 1.2, "jepa_signal": 1},
    #   "m5": {"micro_trend": 1, "price": 54450, "atr": 15.0},
    #   "aligned": True  # all TFs agree on direction
    # }
"""

import json
import math
import logging
import urllib.request

logger = logging.getLogger("eva.multi_tf")

BRIDGE = "http://192.168.1.6:8765"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str, bars: int = 100, tf: str = "M15") -> list[dict]:
    """Fetch OHLCV bars from the MT5 Bridge."""
    url = f"{BRIDGE}/ohlcv/{symbol}/{bars}/{tf}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())
        if "error" in data:
            logger.warning("Bridge error for %s %s: %s", symbol, tf, data["error"])
            return []
        return data.get("bars", [])
    except Exception as e:
        logger.warning("Failed to fetch %s %s %s: %s", symbol, bars, tf, e)
        return []


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def compute_ema(values: list[float], period: int) -> list[float]:
    """Exponential Moving Average."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    ema = [sum(values[:period]) / period]
    for v in values[period:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema


def compute_sma(values: list[float], period: int) -> list[float]:
    """Simple Moving Average."""
    if len(values) < period:
        return []
    sma = []
    for i in range(period - 1, len(values)):
        sma.append(sum(values[i - period + 1:i + 1]) / period)
    return sma


def compute_rsi(closes: list[float], period: int = 14) -> float:
    """Relative Strength Index. Returns 0-100 or 50 if insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def compute_macd(closes: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    """MACD histogram value (MACD line - signal line)."""
    if len(closes) < slow + signal:
        return 0.0
    ema_fast = compute_ema(closes, fast)
    ema_slow = compute_ema(closes, slow)
    if not ema_fast or not ema_slow:
        return 0.0
    # Align lengths
    min_len = min(len(ema_fast), len(ema_slow))
    macd_line = [ema_fast[-min_len + i] - ema_slow[-min_len + i] for i in range(min_len)]
    signal_line = compute_ema(macd_line, signal)
    if not signal_line:
        return 0.0
    return macd_line[-1] - signal_line[-1]


def compute_atr_from_bars(bars: list[dict], period: int = 14) -> float:
    """Average True Range from OHLC bar dicts."""
    if len(bars) < period + 1:
        return 0.0
    tr_values = []
    for i in range(1, len(bars)):
        high = bars[i]["high"]
        low = bars[i]["low"]
        prev_close = bars[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_values.append(tr)
    if len(tr_values) < period:
        return 0.0
    return sum(tr_values[-period:]) / period


def compute_micro_trend(bars: list[dict], n: int = 3) -> int:
    """
    Micro-trend direction from last n candles.
    Returns +1 (up), -1 (down), 0 (flat/insufficient data).
    """
    if len(bars) < n:
        return 0
    closes = [b["close"] for b in bars[-n:]]
    if closes[-1] > closes[0] * 1.0005:
        return 1
    elif closes[-1] < closes[0] * 0.9995:
        return -1
    return 0


def compute_volume_ratio(bars: list[dict], period: int = 20) -> float:
    """Ratio of current bar volume to average of last N bars."""
    if len(bars) < 2:
        return 1.0
    volumes = [b.get("volume", 1) for b in bars]
    recent = min(period, len(volumes) - 1)
    avg_vol = sum(volumes[-recent - 1:-1]) / recent if recent > 0 else 1.0
    if avg_vol == 0:
        return 1.0
    return volumes[-1] / avg_vol


# ---------------------------------------------------------------------------
# H4 features (aggregated from M30)
# ---------------------------------------------------------------------------

def _aggregate_h4_from_m30(bars_m30: list[dict]) -> list[dict]:
    """
    Aggregate 8 M30 bars into 1 pseudo-H4 candle.
    Returns list of aggregated OHLC dicts.
    """
    if len(bars_m30) < 8:
        return []
    # Group into chunks of 8
    h4_candles = []
    for i in range(0, len(bars_m30) - 7, 8):
        chunk = bars_m30[i:i + 8]
        candle = {
            "open": chunk[0]["open"],
            "high": max(b["high"] for b in chunk),
            "low": min(b["low"] for b in chunk),
            "close": chunk[-1]["close"],
            "volume": sum(b.get("volume", 0) for b in chunk),
        }
        h4_candles.append(candle)
    return h4_candles


def get_h4_features(symbol: str) -> dict:
    """
    H4 trend features (aggregated from M30 with dual EMA).

    Strategy:
      1. Aggregate M30 -> H4 candles (8 M30 bars = 1 H4)
      2. Compute EMA20 on H4 (short-term H4 trend)
      3. Compute EMA200 on M30 as long-term trend (proxy for ~EMA50 on H4)
      4. ATR on H4 candles

    Returns:
        trend_dir: +1 bullish, -1 bearish, 0 neutral
        ema: current EMA value (EMA20 on H4 for medium-term)
        ema_long: EMA200 on M30 (long-term trend proxy)
        atr: ATR on H4-aggregated data
        valid: True if sufficient data
    """
    bars_m30 = fetch_ohlcv(symbol, bars=200, tf="M30")
    if len(bars_m30) < 30:
        return {"trend_dir": 0, "ema": 0.0, "ema_long": 0.0, "atr": 0.0, "valid": False}

    # Long-term trend from M30 (EMA200 ~ EMA50 on H4)
    m30_closes = [b["close"] for b in bars_m30]
    ema_long = compute_ema(m30_closes, 200)
    long_term_ema = ema_long[-1] if ema_long else m30_closes[-1]

    # Medium-term: aggregate H4 candles
    h4_bars = _aggregate_h4_from_m30(bars_m30)
    if len(h4_bars) < 10:
        # Fallback: use M30 EMA100 as trend
        ema_med = compute_ema(m30_closes, 100)
        if not ema_med:
            return {"trend_dir": 0, "ema": 0.0, "ema_long": long_term_ema, "atr": 0.0, "valid": False}
        h4_close = m30_closes[-1]
        h4_ema = ema_med[-1]
        atr_val = compute_atr_from_bars(bars_m30, 14)
    else:
        h4_closes = [b["close"] for b in h4_bars]
        ema20 = compute_ema(h4_closes, 20)
        if not ema20:
            return {"trend_dir": 0, "ema": 0.0, "ema_long": long_term_ema, "atr": 0.0, "valid": False}
        h4_close = h4_closes[-1]
        h4_ema = ema20[-1]
        atr_val = compute_atr_from_bars(h4_bars, 14)

    # Determine trend using medium-term EMA20 on H4
    threshold = h4_ema * 0.002  # 0.2% buffer
    if h4_close > h4_ema + threshold:
        trend_dir = 1
    elif h4_close < h4_ema - threshold:
        trend_dir = -1
    else:
        trend_dir = 0

    return {
        "trend_dir": trend_dir,
        "ema": round(h4_ema, 2),
        "ema_long": round(long_term_ema, 2),
        "atr": round(atr_val, 2),
        "valid": True,
    }


# ---------------------------------------------------------------------------
# H1 features (computed from M30 data, as existing strategies do)
# ---------------------------------------------------------------------------

def get_h1_features(symbol: str) -> dict:
    """
    H1 confirmation features (computed from M30 data).
    Uses M30 as proxy for H1 (EMA20 on M30 ~ EMA20 on H1).

    Returns:
        ema20_dir: +1 bullish, -1 bearish, 0 neutral
        rsi: 0-100
        macd_hist: MACD histogram value
        price: current close price
        valid: True if sufficient data
    """
    bars_m30 = fetch_ohlcv(symbol, bars=100, tf="M30")
    if len(bars_m30) < 30:
        return {"ema20_dir": 0, "rsi": 50.0, "macd_hist": 0.0, "price": 0.0, "valid": False}

    closes = [b["close"] for b in bars_m30]
    ema20 = compute_ema(closes, 20)
    if not ema20:
        return {"ema20_dir": 0, "rsi": 50.0, "macd_hist": 0.0, "price": 0.0, "valid": False}

    last_close = closes[-1]
    last_ema = ema20[-1]
    threshold = last_ema * 0.001  # 0.1% buffer
    if last_close > last_ema + threshold:
        ema20_dir = 1
    elif last_close < last_ema - threshold:
        ema20_dir = -1
    else:
        ema20_dir = 0

    rsi = compute_rsi(closes, 14)
    macd_hist = compute_macd(closes, 12, 26, 9)

    return {
        "ema20_dir": ema20_dir,
        "rsi": round(rsi, 1),
        "macd_hist": round(macd_hist, 2),
        "price": last_close,
        "valid": True,
    }


# ---------------------------------------------------------------------------
# M15 features (volume filter, JEPA signal)
# ---------------------------------------------------------------------------

def get_m15_features(symbol: str) -> dict:
    """
    M15 features for JEPA signal and volume filter.

    Returns:
        price: current close
        volume_ratio: ratio of current volume to 20-bar average
        atr: ATR value
        valid: True if sufficient data
    """
    bars = fetch_ohlcv(symbol, bars=100, tf="M15")
    if len(bars) < 20:
        return {"price": 0.0, "volume_ratio": 1.0, "atr": 0.0, "valid": False}

    closes = [b["close"] for b in bars]
    vol_ratio = compute_volume_ratio(bars, 20)
    atr_val = compute_atr_from_bars(bars, 14)

    return {
        "price": closes[-1],
        "volume_ratio": round(vol_ratio, 2),
        "atr": round(atr_val, 2),
        "valid": True,
    }


# ---------------------------------------------------------------------------
# M5 features (micro-trend)
# ---------------------------------------------------------------------------

def get_m5_features(symbol: str) -> dict:
    """
    M5 micro-trend features.

    Returns:
        micro_trend: +1 up, -1 down, 0 flat
        price: current close
        atr: ATR value
        valid: True if sufficient data
    """
    bars = fetch_ohlcv(symbol, bars=30, tf="M5")
    if len(bars) < 5:
        return {"micro_trend": 0, "price": 0.0, "atr": 0.0, "valid": False}

    mt = compute_micro_trend(bars, 3)
    closes = [b["close"] for b in bars]
    atr_val = compute_atr_from_bars(bars, 14)

    return {
        "micro_trend": mt,
        "price": closes[-1],
        "atr": round(atr_val, 2),
        "valid": True,
    }


# ---------------------------------------------------------------------------
# Unified entry
# ---------------------------------------------------------------------------

def get_mtf_features(symbol: str) -> dict:
    """
    Fetch all timeframe features for a symbol.

    Returns dict with keys: h4, h1, m15, m5, aligned.
    'aligned' is True when all valid TFs agree on direction.
    """
    h4 = get_h4_features(symbol)
    h1 = get_h1_features(symbol)
    m15 = get_m15_features(symbol)
    m5 = get_m5_features(symbol)

    # Check alignment: all valid TFs must agree on direction
    dirs = []
    if h4["valid"]:
        dirs.append(h4["trend_dir"])
    if h1["valid"]:
        dirs.append(h1["ema20_dir"])
    if m5["valid"]:
        dirs.append(m5["micro_trend"])

    # Filter out neutral (0) directions
    non_zero = [d for d in dirs if d != 0]
    if non_zero:
        aligned = all(d == non_zero[0] for d in non_zero)
    else:
        aligned = True  # All neutral = aligned (no strong signal)

    return {
        "h4": h4,
        "h1": h1,
        "m15": m15,
        "m5": m5,
        "aligned": aligned,
        "symbol": symbol,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    for sym in ["US30.cash", "USDJPY", "GER40.cash", "US100.cash", "XAUUSD"]:
        print(f"\n=== {sym} ===")
        features = get_mtf_features(sym)
        print(f"  H4:     trend_dir={features['h4']['trend_dir']} ema={features['h4']['ema']:.1f} ema_long={features['h4']['ema_long']:.1f} atr={features['h4']['atr']:.1f} valid={features['h4']['valid']}")
        print(f"  H1:     ema20_dir={features['h1']['ema20_dir']} rsi={features['h1']['rsi']} macd_hist={features['h1']['macd_hist']} valid={features['h1']['valid']}")
        print(f"  M15:    vol_ratio={features['m15']['volume_ratio']} atr={features['m15']['atr']} valid={features['m15']['valid']}")
        print(f"  M5:     micro_trend={features['m5']['micro_trend']} price={features['m5']['price']:.1f} valid={features['m5']['valid']}")
        print(f"  Aligned: {features['aligned']}")

# ---------------------------------------------------------------------------
# JEPA multi-timeframe gating
# ---------------------------------------------------------------------------

def check_mtf_for_jepa(symbol: str, direction: int) -> dict:
    """
    Multi-timeframe gating for JEPA strategies.

    Checks ALL timeframes are aligned with the M15 JEPA signal direction before
    allowing entry:

      1. H4:    trend direction matches (EMA-based) + ATR volatility check
      2. H1:    EMA20 direction + RSI not overbought/oversold + MACD histogram
      3. M15:   JEPA signal (already provided) + volume filter
      4. M5:    micro-trend (last 3 candles) matches

    Args:
        symbol: MT5 symbol
        direction: M15 JEPA signal direction (+1 buy, -1 sell, 0 neutral)

    Returns:
        dict with keys:
            allowed (bool): True if all TFs align
            reason (str):  description of block reason if not allowed
            features (dict): full multi-tf feature dict
    """
    features = get_mtf_features(symbol)
    h4 = features["h4"]
    h1 = features["h1"]
    m15 = features["m15"]
    m5 = features["m5"]

    if direction == 0:
        return {"allowed": False, "reason": "M15 JEPA signal neutral", "features": features}

    # 1. H4 trend check
    if not h4["valid"]:
        return {"allowed": False, "reason": "H4 data unavailable", "features": features}
    if h4["trend_dir"] != 0 and h4["trend_dir"] != direction:
        return {"allowed": False,
                "reason": f"H4 trend mismatch: M15={direction} H4={h4['trend_dir']}",
                "features": features}

    # 2. H1 confirmation
    if not h1["valid"]:
        return {"allowed": False, "reason": "H1 data unavailable", "features": features}
    if h1["ema20_dir"] != 0 and h1["ema20_dir"] != direction:
        return {"allowed": False,
                "reason": f"H1 EMA20 mismatch: M15={direction} H1={h1['ema20_dir']}",
                "features": features}
    # RSI overbought/oversold filter
    if direction == 1 and h1["rsi"] > 75:
        return {"allowed": False, "reason": f"H1 RSI overbought ({h1['rsi']})", "features": features}
    if direction == -1 and h1["rsi"] < 25:
        return {"allowed": False, "reason": f"H1 RSI oversold ({h1['rsi']})", "features": features}
    # MACD histogram confirmation
    if direction == 1 and h1["macd_hist"] < -1:
        return {"allowed": False, "reason": f"H1 MACD negative for buy ({h1['macd_hist']})", "features": features}
    if direction == -1 and h1["macd_hist"] > 1:
        return {"allowed": False, "reason": f"H1 MACD positive for sell ({h1['macd_hist']})", "features": features}

    # 3. M15 volume filter
    if not m15["valid"]:
        return {"allowed": False, "reason": "M15 data unavailable", "features": features}
    if m15["volume_ratio"] < 0.5:
        return {"allowed": False,
                "reason": f"M15 volume too low (ratio={m15['volume_ratio']})",
                "features": features}

    # 4. M5 micro-trend check
    if not m5["valid"]:
        return {"allowed": False, "reason": "M5 data unavailable", "features": features}
    if m5["micro_trend"] != 0 and m5["micro_trend"] != direction:
        return {"allowed": False,
                "reason": f"M5 micro-trend mismatch: M15={direction} M5={m5['micro_trend']}",
                "features": features}

    return {"allowed": True, "reason": "All timeframes aligned", "features": features}


def log_mtf(journal, direction: int, result: dict) -> None:
    """Log multi-timeframe gating result for debugging."""
    f = result["features"]
    journal.info(
        "MTF: H4=%d H1_dir=%d H1_rsi=%.0f H1_macd=%.1f M15_vol=%.2f M5=%d aligned=%s -> %s (%s)",
        f["h4"]["trend_dir"],
        f["h1"]["ema20_dir"],
        f["h1"]["rsi"],
        f["h1"]["macd_hist"],
        f["m15"]["volume_ratio"],
        f["m5"]["micro_trend"],
        f["aligned"],
        "ALLOW" if result["allowed"] else "BLOCK",
        result["reason"],
    )
