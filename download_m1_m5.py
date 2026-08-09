#!/usr/bin/env python3
import os
import json
import urllib.request
from pathlib import Path
import subprocess

DATA_DIR = Path("/home/aza/ftmo_agent/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

symbols = ["US30.cash", "GER40.cash", "US100.cash", "XAUUSD", "USDJPY"]
timeframes = [("M1", 10000), ("M5", 8000), ("M15", 5000)]

print("=== TÉLÉCHARGEMENT HISTORIQUE MULTI-TIMEFRAME (M1, M5, M15) ===")

for sym in symbols:
    for tf, bars_count in timeframes:
        filename = f"{sym}_{tf.lower()}.csv"
        url = f"http://192.168.1.6:8765/ohlcv/{sym}/{bars_count}/{tf}"
        try:
            req = urllib.request.urlopen(url, timeout=10)
            data = json.loads(req.read().decode('utf-8'))
            bars = data.get("bars", [])
            if bars:
                filepath = DATA_DIR / filename
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write("time,open,high,low,close,tick_volume,spread,real_volume\n")
                    for b in bars:
                        f.write(f"{b.get('time',0)},{b.get('open',0)},{b.get('high',0)},{b.get('low',0)},{b.get('close',0)},{b.get('volume',0)},0,0\n")
                print(f"✓ Saved {len(bars)} {tf} candles for {sym} -> {filename}")
        except Exception as e:
            print(f"Error fetching {sym} {tf}: {e}")

print("=== DÉMARRAGE DU MULTI-TIMEFRAME SNIPER TRAINING (M1 & M5) SUR DUAL RTX 3090 ===")
cmd = "cd /home/aza/projects/jepa_eva && PYTHONPATH=. venv/bin/python3 train_jepa.py --symbole US30.cash_m5 --steps 3000 > /tmp/train_m5_gpu.log 2>&1 &"
subprocess.Popen(cmd, shell=True)
print("✓ Multi-Timeframe M1/M5 Training lancé !")
