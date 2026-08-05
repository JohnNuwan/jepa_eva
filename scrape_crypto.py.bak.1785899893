#!/usr/bin/env python3
"""
Scraper de données crypto spot — Binance / Bybit / Bitget
Format de sortie compatible MT5 : time,open,high,low,close,tick_volume,spread,real_volume

Usage :
    PYTHONPATH=. venv/bin/python scrape_crypto.py --pairs BTCUSDT,ETHUSDT,SOLUSDT --timeframe m15
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
import numpy as np

# === Mapping timeframe → Binance interval ===
TIMEFRAMES = {
    "m15": "15m",
    "m5": "5m",
    "h1": "1h",
    "h4": "4h",
    "d1": "1d",
}

# Paires crypto recommandées
PAIRS_CRYPTO = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
]

DATA_DIR = Path(__file__).parent / "data"


def scraper_binance(pair: str, interval: str, limite: int = 1000) -> list[list]:
    """Scrape les chandeliers Binance via l'API REST publique."""
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": pair.upper(),
        "interval": interval,
        "limit": min(limite, 1000),
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def scraper_binance_historique(pair: str, interval: str, total_barres: int = 50000, source: str = "binance") -> list[list]:
    """Scrape l'historique complet par tranches de 1000 (limite API)."""
    toutes = []
    end_time = None

    while len(toutes) < total_barres:
        if source == "bybit":
            interval_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
            inv = interval_map.get(interval, "15")
            params = {
                "category": "spot",
                "symbol": pair.upper(),
                "interval": inv,
                "limit": 200,
            }
            if end_time:
                params["end"] = str(end_time)
            url = "https://api.bybit.com/v5/market/kline"
        else:
            params = {
                "symbol": pair.upper(),
                "interval": interval,
                "limit": 1000,
            }
            if end_time:
                params["endTime"] = end_time
            url = "https://api.binance.com/api/v3/klines"

        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()

        if source == "bybit":
            data = resp.json()
            if data.get("retCode") != 0:
                break
            batch = data["result"]["list"]
            # Bybit renvoie du plus récent au plus ancien
            batch = list(reversed(batch))
        else:
            batch = resp.json()

        if not batch:
            break

        toutes = batch + toutes
        if source == "bybit":
            end_time = int(batch[0][0]) - 1
        else:
            end_time = batch[0][0] - 1

        print(f"   → {len(toutes)} barres récupérées...")

        if len(batch) < (200 if source == "bybit" else 1000):
            break

    return toutes[-total_barres:]


def convertir_en_csv(klines: list[list], pair: str, timeframe: str, chemin_sortie: Path):
    """Convertit les klines Binance → format CSV MT5."""
    with open(chemin_sortie, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])

        for k in klines:
            # Binance kline: [open_time, open, high, low, close, volume, close_time, quote_vol, trades, ...]
            ts_ms = int(k[0])
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")

            o = float(k[1])
            h = float(k[2])
            l = float(k[3])
            c = float(k[4])
            v = float(k[5])  # volume en base asset
            # tick_volume = trades count, real_volume = quote volume
            trades = int(k[8])  # nombre de trades
            quote_vol = float(k[7])  # volume en quote asset

            writer.writerow([time_str, f"{o:.2f}", f"{h:.2f}", f"{l:.2f}", f"{c:.2f}", trades, 1, f"{quote_vol:.2f}"])

    print(f"✅ {pair} {timeframe} → {chemin_sortie} ({len(klines)} barres)")


def scraper_bybit(pair: str, interval: str, limite: int = 200) -> list[list]:
    """Alternative Binance : Bybit API."""
    # Mapping interval Bybit
    interval_map = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}
    inv = interval_map.get(interval, "15")
    url = f"https://api.bybit.com/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": pair.upper(),
        "interval": inv,
        "limit": limite,
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        if data.get("retCode") == 0:
            return data["result"]["list"]
    return []


def main():
    parser = argparse.ArgumentParser(description="Scraper données crypto spot")
    parser.add_argument("--pairs", default="BTCUSDT,ETHUSDT,SOLUSDT",
                        help="Paires séparées par des virgules")
    parser.add_argument("--timeframe", default="m15", choices=TIMEFRAMES.keys())
    parser.add_argument("--barres", type=int, default=50000,
                        help="Nombre de barres historiques (défaut: 50000)")
    parser.add_argument("--source", default="binance", choices=["binance", "bybit"])
    args = parser.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(",")]
    interval = TIMEFRAMES[args.timeframe]
    os.makedirs(DATA_DIR, exist_ok=True)

    for pair in pairs:
        chemin = DATA_DIR / f"{pair}_{args.timeframe}.csv"
        if chemin.exists():
            n = len(open(chemin).readlines()) - 1  # moins header
            print(f"⏭️ {pair} {args.timeframe} existe déjà ({n} barres)")
            continue

        print(f"📡 {pair} {args.timeframe} ({args.barres} barres)...")
        try:
            klines = scraper_binance_historique(pair, interval, args.barres, source=args.source)
            if klines:
                convertir_en_csv(klines, pair, args.timeframe, chemin)
            else:
                # Fallback sur autre source
                autre = "bybit" if args.source == "binance" else "binance"
                print(f"   ⚠️ {args.source} vide, essai {autre}...")
                klines = scraper_binance_historique(pair, interval, min(5000, args.barres), source=autre)
                if klines:
                    convertir_en_csv(klines, pair, args.timeframe, chemin)
                else:
                    print(f"   ❌ {pair}: aucune donnée")
        except Exception as e:
            print(f"   ❌ {pair}: {e}")

    print(f"\n📊 Résumé : {len(pairs)} paire(s) scrappée(s) dans {DATA_DIR}/")
    print("   Pour entraîner :")
    print(f"   PYTHONPATH=. venv/bin/python train_arena.py --symbole BTCUSDT --timeframe {args.timeframe}")


if __name__ == "__main__":
    main()