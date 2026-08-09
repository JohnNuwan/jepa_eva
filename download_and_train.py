#!/usr/bin/env python3
import os
import json
import urllib.request
from pathlib import Path
import subprocess

DATA_DIR = Path("/home/aza/ftmo_agent/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

symbols = [
    ("US30.cash", "US30.cash_m15.csv"),
    ("GER40.cash", "GER40.cash_m15.csv"),
    ("US100.cash", "US100.cash_m15.csv"),
    ("XAUUSD", "XAUUSD_m15.csv"),
    ("USDJPY", "USDJPY_m15.csv")
]

for sym, filename in symbols:
    url = f"http://192.168.1.6:8765/ohlcv/{sym}/5000/M15"
    print(f"Downloading {sym} historical candles from MT5...")
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
            print(f"✓ Saved {len(bars)} candles to {filepath}")
    except Exception as e:
        print(f"Error fetching {sym}: {e}")

print("=== LAUNCHING GPU TRAINING ON DUAL RTX 3090 ===")
cmd = "cd /home/aza/projects/jepa_eva && PYTHONPATH=. venv/bin/python3 train_jepa.py --symbole US30.cash --steps 3000 > /tmp/train_us30_gpu.log 2>&1 &"
subprocess.Popen(cmd, shell=True)
print("✓ GPU Training launched for US30.cash!")
