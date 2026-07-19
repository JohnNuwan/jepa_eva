# JEPA_EVA — Evolving Virtual Asset

Agent de trading évolutif souverain sur cluster local (The Hive, 2× RTX 3090).

Pipeline : encodeur **JEPA** temporel auto-supervisé (PyTorch, GPU 0) → pont
**DLPack** zéro-copie → moteur décisionnel **JAX** (GPU 1) : planificateur
**TD-MPC2** (CEM 5 000 trajectoires) + **arène génétique** (64 agents sous
`jax.vmap`/`jax.jit`). Garde-fous production : sanitizer risque 1 % et
disjoncteur drawdown 4 %/jour.

## État actuel (validé par exécution réelle)

- **Encodeur JEPA pré-entraîné** sur XAUUSD M15 réel (perte 0.129 → 0.0087,
  latents cohérents cos 0.991, pas de collapse), sauvegardé dans `checkpoints_jepa/`.
- **Latents 50 000 barres** pré-calculés (`latents/`).
- **Simulation mark-to-market** réaliste (rendements bornés barre-à-barre).
- **Trades discrets suivis** : win_rate, profit_factor, nb_trades réels par agent.
- **Champions sauvegardés** (poids npz rejouables) dans `registry_arena*/champions/`.
- **Boucle anti-overfitting** (`train_arena_validated.py`) : promotion
  conditionnelle — seuls les champions qui **généralisent** sur le holdout
  (barres 40 000–50 000) sont retenus.
  - Résultat 200 générations : **1/40 champion généralise** — gen4 :
    **+5.06 % holdout, drawdown 0.08 %, profit_factor 17.15, 70 trades**
    (fitness_train faible 0.545 → la sélection par généralisation trouve des
    perles que la fitness train seule rate). Détecte la dérive en
    surapprentissage après ~50 générations (holdout −13 %).

## Structure

| Fichier | Rôle |
|---|---|
| `eva/` | Package modulaire (normalisation, JEPA, pont DLPack, TD-MPC2, arène) |
| `jepa_pipeline.py` | OHLCV → latents 128-dim (GPU 0) |
| `jax_arena.py` | Arène consolidée + CEM + trades discrets + benchmark GPU |
| `action_sanitizer.py` | Risque 1 % + disjoncteur 4 %/jour |
| `donnees_reelles.py` | Chargeur CSV MT5 (8 symboles) |
| `train_jepa.py` | Pré-entraînement JEPA |
| `precompute_latents.py` | Latents de tout l'historique |
| `train_arena.py` | Entraînement arène + sauvegarde poids champions |
| `train_arena_validated.py` | Boucle train→backtest→promotion conditionnelle |
| `backtest_validation.py` | Évaluation holdout d'un champion (anti-overfitting) |
| `main.py` | Orchestrateur boucle de trading |

## Usage

```bash
# 1. Pré-entraîner JEPA (encodeur)
PYTHONPATH=. venv/bin/python train_jepa.py --symbole XAUUSD --steps 2000
# 2. Pré-calculer les latents de tout l'historique
PYTHONPATH=. venv/bin/python precompute_latents.py --symbole XAUUSD
# 3. Entraîner l'arène (simple)
PYTHONPATH=. venv/bin/python train_arena.py --symbole XAUUSD --generations 100
# 3b. Entraîner l'arène AVEC validation holdout (recommandé)
PYTHONPATH=. venv/bin/python train_arena_validated.py --symbole XAUUSD --generations 200
# 4. Backtester un champion sur le holdout
PYTHONPATH=. venv/bin/python backtest_validation.py \
    --champion registry_arena_validated/champions/champion_gen4.npz
# 5. Orchestrateur (boucle de trading)
PYTHONPATH=. venv/bin/python main.py
```

## Prochaines étapes

- **Injecter le score holdout dans la fitness d'évolution** : sélectionner les
  parents sur leur capacité à généraliser (pas seulement le train) pour
  stabiliser l'évolution vers des agents robustes.
- Entraîner le **world model GRU** (prédire H_{t+1} depuis H_t + action) pour
  que le CEM planifie sur des transitions apprises plutôt qu'aléatoires.
- Backtest end-to-end `main.py` en mode rejeu historique avec un champion.
- Multi-symbole (8 symboles dispo : EURUSD, BTCUSD, US30, US500, US100, GER40…).
- Branchement live MT5/cTrader (VM Windows KVM).
