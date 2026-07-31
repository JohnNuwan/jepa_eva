#!/bin/bash
cd /home/aza/projects/jepa_eva
export CUDA_VISIBLE_DEVICES=0
for sym in EURUSD GBPUSD US30.cash US500.cash US100.cash; do
    echo === TRAINING WORLD MODEL: ===
    /home/aza/jepa_eva/venv/bin/python3 train_world_model.py --symbole  --steps 500
    echo === DONE: ===
done
echo === ALL WORLD MODELS TRAINED ===
