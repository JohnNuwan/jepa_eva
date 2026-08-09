#!/bin/bash
cd /home/aza/projects/jepa_eva
echo "=== DÉBUT ENTRAÎNEMENT GLOBAL SUR DUAL RTX 3090 ===" > /tmp/master_gpu_training.log

for sym in "US30.cash" "GER40.cash" "US100.cash" "XAUUSD" "USDJPY"
do
    echo "[$(date)] Entraînement JEPA pour $sym (3000 steps)..." >> /tmp/master_gpu_training.log
    /home/aza/jepa_eva/venv/bin/python3 train_jepa.py --symbole "$sym" --steps 3000 >> /tmp/master_gpu_training.log 2>&1
    echo "[$(date)] ✓ Entraînement $sym Terminé !" >> /tmp/master_gpu_training.log
done

echo "[$(date)] 🎉 TOUS LES ENTRAÎNEMENTS GPU SONT TERMINÉS !" >> /tmp/master_gpu_training.log
