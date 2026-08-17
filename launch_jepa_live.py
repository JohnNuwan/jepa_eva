#!/usr/bin/env python3
"""Lancement JEPA simple — 1 symbole (EURUSD), SL/TP natif, zéro gestion externe."""
import subprocess, sys, os

# Variables d'env
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1 (pas vLLM)
os.environ["EVA_TRADER_DISABLED"] = "1"   # Trader neutralise

cmd = [
    sys.executable, "/home/aza/projects/jepa_eva/main.py",
    "--symbol", "EURUSD",
    "--timeframe", "m15",
    "--volume", "0.01",
    "--max-pos 3
    "--interval", "300",  # 5 min
]

print("Lancement JEPA:", " ".join(cmd))
proc = subprocess.Popen(cmd, cwd="/home/aza/projects/jepa_eva")
try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    print("JEPA arrete")
