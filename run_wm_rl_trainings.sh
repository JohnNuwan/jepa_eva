#!/bin/bash
cd /home/aza/projects/jepa_eva
echo "=== DÉBUT ENTRAÎNEMENT WORLD MODEL RL (JAX/GRU) ON DUAL RTX 3090 ===" > /tmp/wm_rl_training.log

for sym in "US30.cash" "GER40.cash" "US100.cash" "XAUUSD" "USDJPY"
do
    echo "[$(date)] Entraînement World Model RL pour $sym (1000 steps)..." >> /tmp/wm_rl_training.log
    /home/aza/jepa_eva/venv/bin/python3 train_world_model.py --symbole "$sym" --steps 1000 >> /tmp/wm_rl_training.log 2>&1
    echo "[$(date)] ✓ World Model RL $sym Terminé !" >> /tmp/wm_rl_training.log
done

echo "[$(date)] 🎉 TOUS LES ENTRAÎNEMENTS WORLD MODEL RL SONT COMPLÈTEMENT TERMINÉS !" >> /tmp/wm_rl_training.log
