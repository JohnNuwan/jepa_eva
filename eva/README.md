# E.V.A — Evolving Virtual Asset

Pipeline souverain de trading évolutif sur cluster local (The Hive), généré
selon la MASTER-SPECIFICATION. Deux GPU RTX 3090 : PyTorch pour l'encodage
latent (GPU 0), JAX pour la planification et l'évolution (GPU 1), reliés par
DLPack zéro-copie.

## Architecture

```
Flux ticks/OHLCV
    │
    ▼
┌──────────────────────────┐
│ A. DynamicNormalizer     │  rendements multi-horizons [1,2,5,15,30,60],
│    + RunningLayerNorm    │  FFT glissante, écrêtage anti-saturation
└───────────┬──────────────┘
            │ (B, T, 27) — RAM
            ▼
┌──────────────────────────┐
│ A. TimeJEPAEncoder       │  Transformer 1D 4 têtes/3 couches, SDPA/Flash-2
│    + MomentumTarget      │  latent H=128, EMA anti-collapse, Smooth-L1 JEPA
└───────────┬──────────────┘
            │ (B, 128) — GPU 0 (PyTorch)
            ▼
┌──────────────────────────┐
│ B. JAXTransitionBridge   │  DLPack zéro-copie -> GPU 1 (JAX)
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│ B. TDMPC2Planner         │  GRU world model + CEM 5 000 trajectoires
│    JaxGeneticArena       │  vmap 64 agents, fitness Sortino×2−DD%+NetProfit
└───────────┬──────────────┘
            ▼
┌──────────────────────────┐
│ C. ActionSanitizer       │  écrasement lot si risque > 1 % equity
│    DrawdownDisconnecter  │  coupure totale si perte jour ≥ 4 %
└──────────────────────────┘
```

## Modules

| Fichier | Rôle |
|---|---|
| `eva/normalisation.py` | `DynamicNormalizer`, `RunningLayerNorm` |
| `eva/encodeur_jepa.py` | `TimeJEPAEncoder`, `MomentumTarget` (EMA) |
| `eva/pont_jax.py` | `JAXTransitionBridge` DLPack, `pont_defaut()` |
| `eva/planificateur_tdmpc2.py` | `TDMPC2Planner`, world model GRU, CEM |
| `eva/arene_genetique.py` | `JaxGeneticArena`, opérateurs génétiques JAX |
| `eva/assainisseur_actions.py` | `ActionSanitizer`, `LimitesMoneyManagement` |
| `eva/disjoncteur_drawdown.py` | `DrawdownDisconnecter` (garde-fou absolu) |

## Tests

```bash
cd ~/ftmo_agent
PYTHONPATH=. venv/bin/python tests/test_bloc_a.py
PYTHONPATH=. venv/bin/python tests/test_bloc_b.py
PYTHONPATH=. venv/bin/python tests/test_bloc_c.py
PYTHONPATH=. venv/bin/python tests/test_integration_gpu.py
```

Tous validés sur The Hive (2× RTX 3090, torch 2.6+cu124, jax 0.11.0).

## Points de conformité à la spec

- **FlashAttention-2** : `F.scaled_dot_product_attention` forcé dans
  `TimeJEPAEncoder` (Tensor Cores Ampere+).
- **EMA** : θ_target ← 0.999·θ_target + 0.001·θ_online dans `MomentumTarget.maj_ema`.
- **DLPack zéro-copie** : `jax.dlpack.from_dlpack` dans `JAXTransitionBridge.convertir`.
- **CEM 5 000 trajectoires** : `TDMPC2Planner(nb_trajectoires=5000)` par défaut.
- **vmap 64 agents** : `JaxGeneticArena.evaluer_population` vectorisé.
- **Fitness** : `(Sortino × 2) − MaxDrawdown% + NetProfit` dans `_calculer_metriques`.
- **Risque 1 %** : `ActionSanitizer` écrase tout lot dépassant la marge.
- **Disjoncteur 4 %** : `DrawdownDisconnecter` coupe tout et suspend l'IA.

## Références internes

Skill `rl-trading-debugging` — Patterns #15 (validation multi-critères),
#21 (GA + stratégies-règles), #22 (backtest séquentiel ≠ GPU). Le CEM de
`TDMPC2Planner` est de la planification vectorisée (pas un backtest
bar-par-barre), donc compatible avec la parallélisation GPU.
