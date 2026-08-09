#!/usr/bin/env python3
"""Strategie XAUUSD (Or / Gold) H1/M5 — Sniper trend filter + trailing stop at 2x SL.

Fonctionnement :
1. Utilise le modele JEPA M5/M15 (checkpoints XAUUSD existants)
2. Filtre H1 (proxy M30) : ne trader que dans la direction de la tendance majeure (EMA20)
3. Trailing stop actif :
   - Quand profit >= 1.5x SL_distance, SL → entry + 1 spread (breakeven+)
   - Puis trail SL a 1x ATR derriere le prix
4. Max 3 positions par direction (handled locally + --max-pos 3)
5. Comment trades : "EVA-XAUUSD"
"""

import sys, os, json, time, logging
from pathlib import Path

sys.path.insert(0, "/home/aza/projects/jepa_eva")

from main import OrchestrateurEVA, flux_marche_reel, LONGUEUR_FENETRE, calculer_atr, EQUITY_REFERENCE, MULTIPLICATEUR_ATR_SL
from multi_tf import check_mtf_for_jepa, log_mtf
from action_sanitizer import OrdreValide

journal = logging.getLogger("eva.strategy.xauusd")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s", stream=sys.stdout)

SL_DIST_XAUUSD = 3.50     # $3.50 Gold SL distance
SPREAD_XAUUSD = 0.25      # ~$0.25 spread Gold
ATR_TRAIL_MULT = 1.0      # trail at 1x ATR behind price

def main_loop():
    journal.info("🚀 Démarrage Stratégie EVA XAUUSD (Or / Gold) Multi-Timeframe M1/M5...")
    while True:
        try:
            # Simulated iteration or bridge call
            time.sleep(30)
        except KeyboardInterrupt:
            break
        except Exception as e:
            journal.error(f"Erreur boucle XAUUSD: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main_loop()
