#!/usr/bin/env python3
"""Fixed batch training script for all symbols on GPU 1"""
import sys, os, time, subprocess, json
from pathlib import Path

JEPA_DIR = "/home/aza/projects/jepa_eva"
DATA_DIR = "/home/aza/projects/ftmo_agent/data"
VENV = "/home/aza/jepa_eva/venv/bin/python3"
LOG_DIR = "/home/aza/eva-adam-v2/logs"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.chdir(JEPA_DIR)

# Ensure data symlink
if not os.path.islink("data"):
    os.system(f"ln -sf {DATA_DIR} data")

# Symbols to train (sorted by priority)
SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "US30.cash",
    "US500.cash",
    "US100.cash",
    "GER40.cash",
    "BTCUSD",
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "HYPUSDT",
]

existing_jepa = set(f.name for f in Path("checkpoints_jepa").glob("*.pt"))
existing_latents = set(f.name for f in Path("latents").glob("*_latents.npz"))

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()

def run_cmd(cmd, timeout=7200):
    log(f"  → {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    ok = result.returncode == 0
    log(f"  {'✅' if ok else '❌'} exit={result.returncode}")
    if result.stderr:
        for line in result.stderr.strip().split('\n')[-3:]:
            log(f"     {line}")
    return result.returncode

for sym in SYMBOLS:
    sym_safe = sym.replace(".", "_")
    jepa_file = f"jepa_final_{sym_safe}_m15.pt"
    latent_file = f"{sym_safe}_m15_latents.npz"
    csv_file = f"{sym}_m15.csv"
    alt_csv = f"{sym_safe}_m15.csv"
    
    log(f"\n{'='*60}")
    log(f"SYMBOLE: {sym}")
    log(f"{'='*60}")
    
    # Check data
    csv_path = f"{DATA_DIR}/{csv_file}"
    if not os.path.isfile(csv_path):
        csv_path = f"{DATA_DIR}/{alt_csv}"
        if not os.path.isfile(csv_path):
            log(f"  ❌ Données manquantes pour {sym}")
            continue
    
    # Step 1: Train JEPA if needed
    if jepa_file not in existing_jepa:
        log(f"  [1/4] Entraînement JEPA...")
        rc = run_cmd(f"{VENV} train_jepa.py --symbole {sym} --steps 2000")
        if rc != 0:
            continue
        existing_jepa.add(jepa_file)
    else:
        log(f"  [1/4] JEPA déjà existant ✅")
    
    # Step 2: Precompute latents if needed
    if latent_file not in existing_latents:
        log(f"  [2/4] Pré-calcul latents...")
        rc = run_cmd(f"{VENV} precompute_latents.py --symbole {sym}")
        if rc != 0:
            continue
        existing_latents.add(latent_file)
    else:
        log(f"  [2/4] Latents déjà existants ✅")
    
    # Step 3: Train arena
    log(f"  [3/4] Arène génétique (200 générations)...")
    rc = run_cmd(f"{VENV} train_arena.py --symbole {sym} --generations 200")
    if rc != 0:
        log(f"  ⚠ Tentative arène généralisée...")
        run_cmd(f"{VENV} train_arena_generalisee.py --symbole {sym} --generations 200")
    
    # Step 4: Validate
    log(f"  [4/4] Validation holdout...")
    run_cmd(f"{VENV} train_arena_validated.py --symbole {sym} --generations 200")
    
    # Check results
    registry = f"registry_arena_validated/{sym_safe}_registry.jsonl"
    if os.path.isfile(registry):
        with open(registry) as f:
            lines = f.readlines()
        log(f"  ✅ Champions: {len(lines)}")
        for line in lines:
            try:
                c = json.loads(line)
                log(f"     pf={c.get('profit_factor','?')} "
                    f"dd={c.get('drawdown_max','?')} "
                    f"hold={c.get('performance_holdout','?')}")
            except:
                pass
    else:
        log(f"  ⚠ Aucun champion validé")

log(f"\n✅ ENTRAÎNEMENT TERMINÉ")
log(f"Log: {LOG_DIR}/batch_train.log")