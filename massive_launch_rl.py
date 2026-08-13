#!/usr/bin/env python3
"""Lanceur de génération massive de champions RL (PPO, DQN, TD3, SAC).

Génère des configs (variantes × symboles × algorithmes), les répartit en
groupes, et exécute chaque groupe séquentiellement sur GPU 1.

Usage standard (3 workers parallèles sur GPU 1) :
    # Worker 0 : PPO (meilleur pour l'action discrète)
    cd /home/aza/projects/jepa_eva && CUDA_VISIBLE_DEVICES=1 setsid -f bash -c '
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 venv/bin/python3 massive_launch_rl.py \
            --algo ppo --groupe 0 --total 3 > logs_massive/ppo_worker0.log 2>&1'

    # Worker 1 : DQN
    cd /home/aza/projects/jepa_eva && CUDA_VISIBLE_DEVICES=1 setsid -f bash -c '
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 venv/bin/python3 massive_launch_rl.py \
            --algo dqn --groupe 1 --total 3 > logs_massive/dqn_worker1.log 2>&1'

    # Worker 2 : TD3 + SAC (half each)
    cd /home/aza/projects/jepa_eva && CUDA_VISIBLE_DEVICES=1 setsid -f bash -c '
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 venv/bin/python3 massive_launch_rl.py \
            --algo td3 --groupe 2 --total 3 > logs_massive/td3_worker2.log 2>&1'

    # Worker 3 (GPU 0 si libre) : SAC
    cd /home/aza/projects/jepa_eva && CUDA_VISIBLE_DEVICES=0 setsid -f bash -c '
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 venv/bin/python3 massive_launch_rl.py \
            --algo sac --groupe 0 --total 1 > logs_massive/sac_worker3.log 2>&1'

Génération des configs uniquement :
    venv/bin/python3 massive_launch_rl.py --algo ppo --gen-only

Créé par Eva — génération massive de champions RL (août 2026).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Symboles disponibles (latents existants sur TheHive).
SYMBOLES = [
    "US30.cash", "US100.cash", "EURUSD", "USDJPY",
    "GER40.cash", "XAUUSD", "GBPUSD", "US500.cash",
    "BTCUSD", "BTCUSDT", "ETHUSDT", "SOLUSDT",
]

BASE_DIR = Path("registry_rl")
CONFIG_DIR = Path("configs_rl")

# --- Configs par algorithme ------------------------------------------------
CONFIGS_BASE = {
    "symbole": "US30.cash", "timeframe": "m15",
    "frac_train": 0.8, "max_holdout": 1500,
    "np_min_holdout": 0.0, "dd_max_holdout": 5.0,
    "seed": 0,
}

# PPO — discrete actions, parallel envs
VARIANTES_PPO = {
    "baseline": {  # Sortino×2 − DD + NP, 8 envs, 512 steps, 200k steps
        "generations": 200, "steps": 200_000, "segment": 512,
        "nb_envs": 8, "eval_holdout": 10,
        "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2,
        "ent_coef": 0.01, "largeur": 256,
    },
    "explorer": {  # Plus d'exploration (ent_coef élevé, plus d'envs)
        "generations": 200, "steps": 400_000, "segment": 512,
        "nb_envs": 12, "eval_holdout": 10,
        "ent_coef": 0.05, "largeur": 256,
    },
    "exploiteur": {  # Fit plus agressif
        "generations": 300, "steps": 300_000, "segment": 1024,
        "nb_envs": 8, "eval_holdout": 10,
        "ent_coef": 0.005, "clip_ratio": 0.15, "largeur": 256,
    },
    "quick": {  # Rapide pour validation
        "generations": 50, "steps": 50_000, "segment": 256,
        "nb_envs": 4, "eval_holdout": 5,
        "largeur": 128,
    },
}

# DQN — discrete actions, replay buffer
VARIANTES_DQN = {
    "baseline": {
        "steps": 200_000, "segment": 512,
        "eval_holdout": 5000, "double_dqn": True,
        "lr": 1e-3, "gamma": 0.99, "tau": 0.005,
        "eps_debut": 1.0, "eps_fin": 0.02, "eps_decay": 4000,
        "buffer_taille": 100_000, "batch": 64, "largeur": 256,
    },
    "long": {  # Long training, slow epsilon decay
        "steps": 500_000, "segment": 512,
        "eval_holdout": 10000, "double_dqn": True,
        "eps_decay": 10000,
        "buffer_taille": 200_000, "batch": 64, "largeur": 256,
    },
    "rapide": {  # Test rapide
        "steps": 50_000, "segment": 256,
        "eval_holdout": 2500,
        "eps_decay": 1000, "buffer_taille": 20_000,
        "largeur": 128,
    },
}

# TD3 — continuous actions, deterministic policy
VARIANTES_TD3 = {
    "baseline": {
        "steps": 200_000, "segment": 512,
        "eval_holdout": 5000,
        "lr_acteur": 3e-4, "lr_critique": 1e-3, "gamma": 0.99,
        "tau": 0.005, "bruit_std": 0.1, "bruit_cible": 0.2,
        "clip_cible": 0.5, "delay_acteur": 2,
        "buffer_taille": 100_000, "batch": 64, "largeur": 256,
    },
    "agressif": {  # Plus d'exploration, plus long
        "steps": 400_000, "segment": 512,
        "eval_holdout": 10000,
        "bruit_std": 0.2, "bruit_cible": 0.3,
        "buffer_taille": 200_000, "largeur": 256,
    },
    "rapide": {
        "steps": 50_000, "segment": 256,
        "eval_holdout": 2500,
        "buffer_taille": 20_000, "largeur": 128,
    },
}

# SAC — continuous actions, stochastic policy, entropy max
VARIANTES_SAC = {
    "baseline": {
        "steps": 200_000, "segment": 512,
        "eval_holdout": 5000,
        "lr": 3e-4, "gamma": 0.99, "tau": 0.005,
        "alpha": 0.2, "alpha_auto": True,
        "buffer_taille": 100_000, "batch": 64, "largeur": 256,
    },
    "explorer": {  # Plus d'entropie
        "steps": 300_000, "segment": 512,
        "eval_holdout": 10000,
        "alpha": 0.5, "alpha_auto": True,
        "buffer_taille": 200_000, "batch": 64, "largeur": 256,
    },
    "rapide": {
        "steps": 50_000, "segment": 256,
        "eval_holdout": 2500,
        "buffer_taille": 20_000, "largeur": 128,
    },
}

VARIANTES = {
    "ppo": VARIANTES_PPO,
    "dqn": VARIANTES_DQN,
    "td3": VARIANTES_TD3,
    "sac": VARIANTES_SAC,
}

MODULES = {
    "ppo": "rl_ppo",
    "dqn": "rl_dqn",
    "td3": "rl_td3",
    "sac": "rl_sac",
}


def generer_configs(algo: str) -> list[dict]:
    """Génère toutes les configs pour un algorithme et les écrit en JSON.

    Args:
        algo: Nom de l'algo (ppo, dqn, td3, sac).

    Returns:
        Liste des configs générées.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    variantes = VARIANTES.get(algo, {})
    toutes = []
    for sym in SYMBOLES:
        for tag, params in variantes.items():
            cfg = {
                **CONFIGS_BASE,
                "symbole": sym,
                "id": f"{algo}_{sym.replace('.', '_')}_{tag}",
                **params,
            }
            chemin = CONFIG_DIR / f"{cfg['id']}.json"
            chemin.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            toutes.append(cfg)
    return toutes


def lancer(configs: list[dict], algo: str) -> None:
    """Exécute les runs séquentiellement.

    Args:
        configs: Liste des configs à exécuter.
        algo: Nom de l'algo (ppo, dqn, td3, sac).
    """
    os.makedirs("logs_massive", exist_ok=True)
    module_name = MODULES.get(algo, f"rl_{algo}")
    total = len(configs)
    ok = 0
    echecs = []
    for i, cfg in enumerate(configs, 1):
        run_id = cfg["id"]
        if (BASE_DIR / run_id / "run_meta.json").is_file():
            print(f"[{i}/{total}] SKIP {run_id} (déjà fait)")
            ok += 1
            continue
        print(f"[{i}/{total}] RUN {run_id} ({algo.upper()}) ...", flush=True)
        t0 = time.time()
        try:
            # Import dynamique du module de l'algo.
            import importlib
            mod = importlib.import_module(module_name)
            resume = mod.lancer_run(cfg)
            ok += 1
            dur = resume.get("duree_s", time.time() - t0)
            print(f"  -> {resume['nb_champions']} champions en {dur:.0f}s (best={resume.get('meilleur_holdout', 'N/A')})", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            echecs.append((run_id, str(e)))
            print(f"  -> ÉCHEC {run_id}: {e}", flush=True)
        print(f"  ({time.time()-t0:.0f}s)", flush=True)
    print(f"TERMINÉ {algo} : {ok}/{total} OK | échecs: {echecs}", flush=True)


def main() -> None:
    """Point d'entrée CLI."""
    p = argparse.ArgumentParser(description="Lanceur massif de champions RL")
    p.add_argument("--algo", required=True, choices=["ppo", "dqn", "td3", "sac"],
                   help="Algorithme RL à lancer")
    p.add_argument("--groupe", type=int, default=0, help="Index du groupe (0-indexed)")
    p.add_argument("--total", type=int, default=1, help="Nombre de groupes")
    p.add_argument("--gen-only", action="store_true", help="Génère les configs puis sort")
    p.add_argument("--configs", default=None, help="Fichier JSON liste de configs (optionnel)")
    args = p.parse_args()

    toutes = generer_configs(args.algo)
    if args.gen_only:
        print(f"{len(toutes)} configs {args.algo} générées dans {CONFIG_DIR}/")
        return

    if args.configs:
        raw = json.loads(Path(args.configs).read_text(encoding="utf-8"))
        liste = raw if isinstance(raw, list) else [raw]
        groupe = [c for i, c in enumerate(liste) if i % args.total == args.groupe]
    else:
        groupe = [c for i, c in enumerate(toutes) if i % args.total == args.groupe]

    print(f"Lancement {args.algo} : groupe {args.groupe}/{args.total} ({len(groupe)} runs)")
    lancer(groupe, args.algo)


if __name__ == "__main__":
    main()