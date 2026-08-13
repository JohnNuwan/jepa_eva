# RL Trading E.V.A — Algorithmes de Reinforcement Learning pour JEPA

## Vue d'ensemble

4 algorithmes RL autonomes pour le trading M15 sur les latents JEPA, implémentés
en PyTorch 2.6+ CUDA. Chaque module est un fichier `.py` autonome (sans dépendances
externes au-delà de numpy, pandas, torch).

| Algo | Type | Actions | Fichier | Référence |
|------|------|---------|---------|-----------|
| **PPO** | Actor-Critic | Discret 3 (SHORT/FLAT/LONG) | `rl_ppo.py` | Schulman et al. 2017 |
| **DQN** | Q-Learning | Discret 3 (SHORT/FLAT/LONG) | `rl_dqn.py` | Mnih et al. 2015 |
| **TD3** | Actor-Critic (continu) | Continu [-1, +1] | `rl_td3.py` | Fujimoto et al. 2018 |
| **SAC** | Actor-Critic (continu, stochastique) | Continu [-1, +1] | `rl_sac.py` | Haarnoja et al. 2018 |

## Architecture commune

### Environnement

Chaque module réimplémente un environnement de trading `EnvironnementTrading` dont
la récompense par barre M15 est **strictement identique** à la simulation de
`jax_arena._pas_simulation` :

```
retour_net = position × (Δprix / prix) × LEVIER − |Δposition| × COUT_TRANSACTION × LEVIER
```

- **Observation** : concaténation de [latent JEPA (128), position courante (1), fenêtre des 16 derniers retours (16)] → **145 dimensions**
- **Actions discrètes** : {0: SHORT → -1, 1: FLAT → 0, 2: LONG → +1}
- **Actions continues** : a ∈ [-1, 1] = position cible
- **Métriques finales** : fitness = Sortino×2 − MaxDrawdown + NetProfit (%) + win_rate, profit_factor, nb_trades

### Format champion (.npz)

Compatible avec les conventions de `champion_factory.py` :

| Clé | Description |
|-----|-------------|
| `p0..pN` | Poids aplatis du réseau (state_dict ordonné) |
| `fitness` | Score Sortino×2 − DD + NP |
| `net_profit` | Profit net en % du capital |
| `max_drawdown` | Drawdown max en % |
| `sortino` | Ratio de Sortino |
| `win_rate` | Taux de trades gagnants (%) |
| `profit_factor` | Profit brut / perte brute |
| `nb_trades` | Nombre de trades fermés |
| `algo` | Nom de l'algorithme (méta-donnée) |
| `symbole` | Symbole (méta-donnée) |

### Sorties

Chaque run crée dans `registry_rl/<run_id>/` :
- `run_meta.json` — config + métriques finales
- `registry.jsonl` — champions validés (holdout) avec métriques
- `candidates.jsonl` — tous les records intermédiaires
- `champions/champion_final.npz` — meilleur champion final
- `champions/champion_p{steps}.npz` — champions par génération/pas

## Utilisation

### Test rapide (∼10s, vérifie absence de crash)

```bash
# GPU 1 (recommandé)
cd /home/aza/projects/jepa_eva

CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_ppo.py --test
CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_dqn.py --test
CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_td3.py --test
CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_sac.py --test

# CPU (si GPU saturé)
venv/bin/python3 rl_ppo.py --test
```

### Entraînement simple

```bash
# PPO — recommandé pour commencer (discret, rapide, robuste)
CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_ppo.py \
    --symbole US30.cash --run-id ppo_us30_v1 --steps 200000 --segment 512

# DQN — discret, apprend plus lentement mais plus stable
CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_dqn.py \
    --symbole US30.cash --run-id dqn_us30_v1 --steps 200000 --segment 512

# TD3 — continu, recherche de champion la plus rapide (tool TD3: +1.33 fitness en 20k steps)
CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_td3.py \
    --symbole US30.cash --run-id td3_us30_v1 --steps 200000 --segment 512

# SAC — continu, stochastique, exploration naturelle
CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_sac.py \
    --symbole US30.cash --run-id sac_us30_v1 --steps 200000 --segment 512
```

### Via fichier de config JSON

```bash
# Générer une config
mkdir -p configs_rl
cat > configs_rl/ppo_US30.json << 'EOF'
{
  "id": "ppo_us30_v1",
  "symbole": "US30.cash",
  "timeframe": "m15",
  "generations": 200,
  "steps": 200000,
  "segment": 512,
  "nb_envs": 8,
  "eval_holdout": 10,
  "lr": 0.0003,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "clip_ratio": 0.2,
  "ent_coef": 0.01
}
EOF

CUDA_VISIBLE_DEVICES=1 venv/bin/python3 rl_ppo.py --config configs_rl/ppo_US30.json
```

### Lancement massif

```bash
# Générer toutes les configs
venv/bin/python3 massive_launch_rl.py --algo ppo --gen-only

# Lancer 1/3 des runs PPO (worker 0)
CUDA_VISIBLE_DEVICES=1 venv/bin/python3 massive_launch_rl.py \
    --algo ppo --groupe 0 --total 3

# Lancer 1/3 des runs DQN (worker 1)
CUDA_VISIBLE_DEVICES=1 venv/bin/python3 massive_launch_rl.py \
    --algo dqn --groupe 0 --total 1
```

## Guide par algorithme

### PPO (`rl_ppo.py`)

**Meilleur choix pour commencer.** Apprentissage stable, parallélisé (8 envs),
bon compromis vitesse/qualité.

| Hyperparamètre | Défaut | Description |
|----------------|--------|-------------|
| `lr` | 3e-4 | Taux d'apprentissage |
| `gamma` | 0.99 | Facteur d'actualisation |
| `gae_lambda` | 0.95 | λ de GAE |
| `clip_ratio` | 0.2 | ε de clipping |
| `ent_coef` | 0.01 | Bonus d'entropie |
| `nb_envs` | 8 | Environnements parallèles |
| `segment` | 512 | Pas par rollout |
| `largeur` | 256 | Neurones par couche |

**Résultats attendus** : ∼200k steps = ∼50 updates × 8 envs. Compter 2-5 min sur
GPU 1. Un champion valide apparaît généralement après 80-150 updates.

### DQN (`rl_dqn.py`)

**Apprentissage lent mais stable.** Convient pour les budgets d'exploration longs.

| Hyperparamètre | Défaut | Description |
|----------------|--------|-------------|
| `lr` | 1e-3 | Taux d'apprentissage |
| `gamma` | 0.99 | Facteur d'actualisation |
| `tau` | 0.005 | Copie douce du réseau cible |
| `eps_debut` | 1.0 | Epsilon initial (100% exploration) |
| `eps_fin` | 0.02 | Epsilon final |
| `eps_decay` | 4000 | Pas avant décroissance complète |
| `double_dqn` | True | Double DQN activé |
| `buffer_taille` | 100000 | Capacité du replay buffer |

**Résultats attendus** : ∼200k steps = 200k transitions. Compter 1-3 min sur GPU 1.
L'apprentissage est lent les premiers 50k pas (exploration pure).

### TD3 (`rl_td3.py`)

**Recherche de champion la plus rapide** (a trouvé un champion +1.33 fitness en
seulement 20k steps). Actions continues, acteur déterministe.

| Hyperparamètre | Défaut | Description |
|----------------|--------|-------------|
| `lr_acteur` | 3e-4 | Taux de l'acteur (bas) |
| `lr_critique` | 1e-3 | Taux des critiques (haut) |
| `gamma` | 0.99 | Facteur d'actualisation |
| `tau` | 0.005 | Copie douce |
| `bruit_std` | 0.1 | Bruit d'exploration |
| `bruit_cible` | 0.2 | Lissage de cible |
| `delay_acteur` | 2 | Mise à jour retardée de l'acteur |

**Résultats attendus** : ∼200k steps = 2-4 min. TD3 converge souvent plus vite
que PPO, mais le champion final est un acteur déterministe (pas de stochastique
pour l'exploration en production).

### SAC (`rl_sac.py`)

**Le plus théoriquement solide** (entropie maximale, exploration naturelle).
Mais le plus lent en pratique (243s pour 20k steps vs 9s pour PPO).

| Hyperparamètre | Défaut | Description |
|----------------|--------|-------------|
| `lr` | 3e-4 | Taux d'apprentissage (commun) |
| `gamma` | 0.99 | Facteur d'actualisation |
| `tau` | 0.005 | Copie douce |
| `alpha` | 0.2 | Température initiale |
| `alpha_auto` | True | Ajustement automatique de α |
| `entropie_cible` | -1 | Cible d'entropie (auto si None) |

**Résultats attendus** : ∼200k steps = 8-15 min. SAC est lent à cause de la
ré-paramétrisation stochastique et des 3 réseaux (politique + 2 critiques).
Avantage : exploration naturelle même après convergence.

## Comparaison des performances

Tests sur GPU 1 (RTX 3090, US30.cash, 20k steps) :

| Algo | Durée | Champions | Meilleur fitness | Convergence |
|------|-------|-----------|-----------------|-------------|
| **PPO** | 9s | 0 | -inf (5 gen) | Très rapide par itération |
| **DQN** | 64s | 0 | -inf (20k steps) | Lent mais stable |
| **TD3** | 116s | 1 | **+1.33** | Champion dès 10k steps |
| **SAC** | 243s | 2 | -0.12 | Le plus lent |

## Structure des fichiers

```
/home/aza/projects/jepa_eva/
├── rl_ppo.py                 # PPO discret
├── rl_dqn.py                 # DQN discret
├── rl_td3.py                 # TD3 continu
├── rl_sac.py                 # SAC continu
├── massive_launch_rl.py      # Lanceur massif multi-algo
├── configs_rl/               # Configs JSON générées
│   ├── ppo_US30_cash_baseline.json
│   ├── dqn_EURUSD_baseline.json
│   └── ...
├── registry_rl/              # Résultats des runs
│   ├── ppo_us30_v1/
│   │   ├── run_meta.json
│   │   ├── registry.jsonl
│   │   ├── candidates.jsonl
│   │   └── champions/
│   │       ├── champion_final.npz
│   │       └── champion_up10.npz
│   ├── td3_us30_v1/
│   └── ...
└── logs_massive/             # Logs des lancements massifs
    ├── ppo_worker0.log
    └── ...
```

## Liens avec l'infrastructure existante

- **Données** : utilise les mêmes latents JEPA que `jax_arena` et `champion_factory`
  (`latents/{sym}_m15_latents.npz`, clés `prix` + `latents`)
- **Récompense** : strictement identique à `jax_arena._pas_simulation`
- **Métriques** : fitness = Sortino×2 − DD + NP (identique à `champion_factory`)
- **Format champion** : `.npz` avec clés `p0..pN` + métriques (compatible avec
  `backtest_validation.charger_champion` pour le format, mais pas la structure
  des poids — RL champions utilisent leur propre `charger_champion()`)
- **Lancement massif** : mêmes conventions que `massive_launch.py` (groupes,
  configs JSON, skip si déjà fait)