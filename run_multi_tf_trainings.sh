#!/bin/bash
cd /home/aza/projects/jepa_eva
echo "=== DÉBUT ENTRAÎNEMENT MULTI-TIMEFRAME (M1, M5, M15) SUR DUAL RTX 3090 ===" > /tmp/multi_tf_gpu_training.log

for sym in "US30.cash" "GER40.cash" "US100.cash" "XAUUSD" "USDJPY"
do
    for tf in "M1" "M5" "M15"
    do
        echo "[$(date)] Entraînement JEPA pour $sym ($tf) — 3000 steps..." >> /tmp/multi_tf_gpu_training.log
        /home/aza/jepa_eva/venv/bin/python3 train_jepa.py --symbole "$sym" --timeframe "$tf" --steps 3000 >> /tmp/multi_tf_gpu_training.log 2>&1
        echo "[$(date)] ✓ Entraînement $sym ($tf) Terminé !" >> /tmp/multi_tf_gpu_training.log
    done
done

echo "[$(date)] 🎉 TOUS LES ENTRAÎNEMENTS MULTI-TIMEFRAME M1/M5/M15 GPU SONT TERMINÉS !" >> /tmp/multi_tf_gpu_training.log
