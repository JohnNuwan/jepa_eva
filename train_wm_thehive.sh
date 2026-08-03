#!/bin/bash
LOG=/home/aza/projects/jepa_eva/wm_thehive.log
cd /home/aza/projects/jepa_eva

echo "[$(date)] === WORLD MODEL TRAINING ON THEHIVE GPU 0 ===" > $LOG

SYMBOLS=$(ls latents/*_m15_latents.npz 2>/dev/null | sed 's/latents\///' | sed 's/_m15_latents.npz//')

for SYM in $SYMBOLS; do
    if [ -f "checkpoints_wm/world_model_${SYM}_m15.npz" ]; then
        echo "[$(date)] SKIP $SYM (WM exists)" >> $LOG
        continue
    fi
    echo "[$(date)] Training WM for $SYM..." >> $LOG
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0         /home/aza/projects/jepa_eva/venv/bin/python3 train_world_model.py         --symbole "$SYM" --steps 500 >> $LOG 2>&1
    if [ $? -eq 0 ]; then
        echo "[$(date)] ✅ $SYM done" >> $LOG
    else
        echo "[$(date)] ❌ $SYM failed" >> $LOG
    fi
done

echo "[$(date)] === WM TRAINING COMPLETE ===" >> $LOG
echo "[$(date)] Total WM: $(ls checkpoints_wm/world_model_*.npz 2>/dev/null | wc -l)" >> $LOG
