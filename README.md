# JEPA_EVA — Evolving Virtual Asset

Agent de trading évolutif souverain sur cluster local (The Hive, 2× RTX 3090).

Pipeline : encodeur **JEPA** temporel auto-supervisé (PyTorch, GPU 0) → pont
**DLPack** zéro-copie → moteur décisionnel **JAX** (GPU 1) : planificateur
**TD-MPC2** (CEM 5 000 trajectoires) + **arène génétique** (64 agents sous
`jax.vmap`/`jax.jit`). Garde-fous production : sanitizer risque 1 % et
disjoncteur drawdown 4 %/jour.

## État actuel

- Encodeur JEPA **pré-entraîné** sur XAUUSD M15 réel (perte 0.129 → 0.0087),
  latents cohérents (cos 0.991), sauvegardé dans `checkpoints_jepa/`.
- Latents 50 000 barres pré-calculés (`latents/`).
- Arène génétique entraînée 100 générations (record fitness 3.41, 7 champions
  promus tracés dans `registry_arena/`).
- Simulation mark-to-market réaliste (corrigée — plus d'explosion de levier).

## Structure

| Fichier | Rôle |
|---|---|
| `eva/` | Package modulaire (normalisation, JEPA, pont DLPack, TD-MPC2, arène) |
| `jepa_pipeline.py` | OHLCV → latents 128-dim (GPU 0) |
| `jax_arena.py` | Arène consolidée + CEM + benchmark GPU |
| `action_sanitizer.py` | Risque 1 % + disjoncteur 4 % |
| `donnees_reelles.py` | Chargeur CSV MT5 (8 symboles) |
| `train_jepa.py` | Pré-entraînement JEPA |
| `precompute_latents.py` | Latents de tout l'historique |
| `train_arena.py` | Entraînement arène + champion registry |
| `main.py` | Orchestrateur boucle de trading |

## Usage

```bash
# Pré-entraîner JEPA
PYTHONPATH=. venv/bin/python train_jepa.py --symbole XAUUSD --steps 2000
# Pré-calculer les latents
PYTHONPATH=. venv/bin/python precompute_latents.py --symbole XAUUSD
# Entraîner l'arène
PYTHONPATH=. venv/bin/python train_arena.py --symbole XAUUSD --generations 100
# Orchestrateur (boucle)
PYTHONPATH=. venv/bin/python main.py
```

## Prochaines étapes

- Sauvegarder les **poids** du champion (pas seulement les métriques).
- Suivi des trades discrets (win_rate, profit_factor) → promotion multi-critères.
- Backtest de validation sur le holdout (barres 40 000–50 000).
- Entraînement du world model GRU pour la planification CEM.
- Branchement live MT5/cTrader.
