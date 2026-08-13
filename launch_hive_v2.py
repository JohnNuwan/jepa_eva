#!/usr/bin/env python3
import os, sys, json, time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

CONFIG_DIR = Path("/home/aza/projects/jepa_eva/configs_massive_v2")
GROUP = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 3

configs = []
for f in sorted(CONFIG_DIR.glob("*.json")):
    configs.append(json.loads(f.read_text()))

groupe = [c for i, c in enumerate(configs) if i % TOTAL == GROUP]
print(f"[Groupe {GROUP}/{TOTAL}] {len(groupe)} configs a lancer", flush=True)

sys.path.insert(0, "/home/aza/projects/jepa_eva")
from champion_factory import lancer_run

for i, cfg in enumerate(groupe, 1):
    run_id = cfg["id"]
    base = Path("registry_massive")
    meta = base / run_id / "run_meta.json"
    if meta.is_file():
        print(f"  [{i}/{len(groupe)}] SKIP {run_id}", flush=True)
        continue
    print(f"  [{i}/{len(groupe)}] RUN {run_id} ...", flush=True)
    t0 = time.time()
    try:
        res = lancer_run(cfg)
        print(f"    -> {res.get('nb_champions',0)} champions en {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"    FAIL: {e}", flush=True)

print(f"[Groupe {GROUP}/{TOTAL}] TERMINE", flush=True)
