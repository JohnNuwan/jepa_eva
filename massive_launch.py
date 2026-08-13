#!/usr/bin/env python3
"""Lanceur de génération massive de champions.

Génère des configs (variantes × symboles), les répartit en groupes, et
exécute chaque groupe séquentiellement avec un seul process JAX (économie
de mémoire GPU). À lancer en 2-3 instances parallèles :
    setsid -f bash -c 'cd /home/debia/jepa_eva && PYTHONPATH=. XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 .venv/bin/python3 massive_launch.py --groupe 0 --total 3 > /home/debia/jepa_eva/logs_massive/worker0.log 2>&1'

Créé par Eva — génération massive de champions (août 2026).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path("registry_massive")
CONFIG_DIR = Path("configs_massive")

# Symboles prioritaires (données M15 disponibles, SPCX absent des CSV).
SYMBOLES = [
    "US30.cash", "US100.cash", "EURUSD", "USDJPY",
    "GER40.cash", "XAUUSD", "GBPUSD",
]

# Variantes de stratégie (recherche : fitness, coût, capacité, évolution).
def variantes(sym: str) -> list[dict]:
    base = {
        "symbole": sym, "timeframe": "m15",
        "generations": 250, "segment": 512,
        "taille_pop": 64, "nb_elites": 16,
        "taux_mutation": 0.1, "sigma_mutation": 0.02,
        "dim_cache": 256, "dim_action": 8,
        "cout_transaction": 0.0002,
        "fitness": {"sortino": 2.0, "sharpe": 0.0, "dd": 1.0, "np": 1.0, "pf": 0.0, "wr": 0.0},
        "frac_train": 0.8, "eval_holdout": 10,
        "walkforward": 1, "max_holdout": 1500,
        "np_min_holdout": 0.0, "dd_max_holdout": 5.0,
        "seed": 0,
    }

    def mk(tag: str, **over) -> dict:
        c = {**base, **over}
        c["id"] = f"{sym.replace('.', '_')}_{tag}"
        return c

    return [
        # 1. Baseline (référence Sortino×2 − DD + NP)
        mk("baseline"),
        # 2. Fitness Sharpe (momentum tolerant)
        mk("sharpe", fitness={"sortino": 0.0, "sharpe": 2.0, "dd": 1.0, "np": 1.0, "pf": 0.0, "wr": 0.0}),
        # 3. Fitness Calmar-like (NP/DD agressif sur rentabilité nette)
        mk("calmar", fitness={"sortino": 0.5, "sharpe": 0.0, "dd": 2.0, "np": 2.0, "pf": 0.0, "wr": 0.0}),
        # 4. Fitness profit-factor (qualité des trades)
        mk("pf", fitness={"sortino": 1.0, "sharpe": 0.0, "dd": 1.0, "np": 1.0, "pf": 2.0, "wr": 0.0}),
        # 5. Fitness win-rate (conservateur)
        mk("wr", fitness={"sortino": 1.0, "sharpe": 0.0, "dd": 1.5, "np": 0.5, "pf": 0.0, "wr": 1.5}),
        # 6. Coût élevé → trades sélectifs
        mk("highcost", cout_transaction=0.001, fitness={"sortino": 2.0, "sharpe": 0.0, "dd": 1.0, "np": 1.0, "pf": 0.5, "wr": 0.0}),
        # 7. Coût faible → trading actif
        mk("lowcost", cout_transaction=0.00002, fitness={"sortino": 1.5, "sharpe": 0.0, "dd": 1.0, "np": 1.5, "pf": 0.0, "wr": 0.0}),
        # 8. GRU large (capacité)
        mk("biggru", dim_cache=512),
        # 9. GRU compact (régularisation implicite)
        mk("smallgru", dim_cache=128),
        # 10. Segment court (régimes rapides)
        mk("shortseg", segment=256),
        # 11. Segment long (contextualisation)
        mk("longseg", segment=1024),
        # 12. Population large + mutation forte (exploration)
        mk("explore", taille_pop=128, nb_elites=24, taux_mutation=0.2, sigma_mutation=0.05),
        # 13. Population petite + élites nombreuses (exploitation)
        mk("exploit", taille_pop=32, nb_elites=20, taux_mutation=0.05, sigma_mutation=0.01),
        # 14. Walk-forward 3 folds (généralisation maximale)
        mk("wf3", walkforward=3, eval_holdout=15),
        # 15. Seed différente (diversité génétique)
        mk("seed7", seed=7),
    ]


def generer_configs() -> list[dict]:
    """Génère toutes les configs et les écrit en fichiers JSON."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    toutes = []
    for sym in SYMBOLES:
        for v in variantes(sym):
            chemin = CONFIG_DIR / f"{v['id']}.json"
            chemin.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")
            toutes.append(v)
    return toutes


def lancer(configs: list[dict]) -> None:
    """Exécute les runs séquentiellement pour ce groupe."""
    os.makedirs("logs_massive", exist_ok=True)
    total = len(configs)
    ok = 0
    echecs = []
    for i, cfg in enumerate(configs, 1):
        run_id = cfg["id"]
        if (Path("registry_massive") / run_id / "run_meta.json").is_file():
            print(f"[{i}/{total}] SKIP {run_id} (déjà fait)")
            ok += 1
            continue
        print(f"[{i}/{total}] RUN {run_id} ...", flush=True)
        t0 = time.time()
        try:
            from champion_factory import lancer_run
            resume = lancer_run(cfg)
            ok += 1
            print(f"  -> {resume['nb_champions']} champions en {resume['duree_s']:.0f}s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            echecs.append((run_id, str(e)))
            print(f"  -> ÉCHEC {run_id}: {e}", flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
    print(f"TERMINÉ : {ok}/{total} OK | échecs: {echecs}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--groupe", type=int, default=0, help="index du groupe")
    p.add_argument("--total", type=int, default=3, help="nombre de groupes")
    p.add_argument("--gen-only", action="store_true", help="génère les configs puis sort")
    p.add_argument("--configs", default=None, help="fichier JSON liste de configs (optionnel)")
    args = p.parse_args()

    toutes = generer_configs()
    if args.gen_only:
        print(f"{len(toutes)} configs générées dans {CONFIG_DIR}/")
        return

    if args.configs:
        from champion_factory import lancer_run
        liste = json.loads(Path(args.configs).read_text(encoding="utf-8"))
        # Répartir équitablement
        groupe = [c for i, c in enumerate(liste) if i % args.total == args.groupe]
        lancer(groupe)
    else:
        groupe = [c for i, c in enumerate(toutes) if i % args.total == args.groupe]
        lancer(groupe)


if __name__ == "__main__":
    main()
