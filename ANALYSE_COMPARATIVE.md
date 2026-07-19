# Analyse comparative — E.V.A vs approches précédentes

Document d'évaluation honnête du projet E.V.A par rapport aux 6 approches
antérieures (DreamerV3, PPO, ES/LSTM V5→V7d, EVO-ARENA GA), basée sur les
patterns documentés dans `rl-trading-debugging` (Patterns #1–#22).

## Verdict en une phrase

E.V.A est le projet le plus **solide méthodologiquement** — premier à prouver
la généralisation sur données jamais vues — mais il n'est **pas encore prouvé
rentable en conditions réelles** (live FTMO).

## Comparaison mesurable

| Critère | Avant (6 échecs) | E.V.A |
|---|---|---|
| Approche | DreamerV3, PPO, ES/LSTM | JEPA + GA + TD-MPC2 |
| Convergence | collapse (reward→moyenne, entropie→0) | perte JEPA 0.129 → 0.0087 |
| Validation | PnL-seul, best.pt écrasé | holdout + registry + multi-critères |
| Généralisation | jamais testée (pas de holdout) | **prouvée 8/8 symboles** |
| Surapprentissage | invisible | **détecté et rejeté** (gen9/23) |
| Temps/boucle | ~100 s/gen (V7d), 47 s (V7c) | ~4 s/gen, CEM 7 ms |
| Trades réels | 0 trades master (collapse HOLD) | champion 70–82 trades, pf 13–17 |

## Ce qui est objectivement mieux

1. **Premier projet à PROUVER la généralisation.** Avant, `best.pt` était
   sauvegardé sur `val_pnl > 0` — un coup de chance sur 500 steps. Le champion
   gen0 d'E.V.A a été évalué sur 8 symboles JAMAIS vus et reste rentable sur
   les 8 (XAUUSD +8.11 %, BTCUSD +13.99 %, US100 +3.88 %, GER40 +2.93 %…).

2. **Détection automatique de la surapprentissage.** Le backtest holdout a
   immédiatement rejeté gen23 (+2 % train → −8 % holdout) et gen9. Sur les
   projets précédents, ces modèles auraient été déployés en live et auraient
   perdu de l'argent sans explication.

3. **Pas de collapse.** Tous les échecs précédents (DreamerV3 reward→moyenne,
   PPO entropie→0, ES master 0 trades) venaient du même problème : les NN
   profonds surapprennent le bruit M15. E.V.A contourne ça : JEPA apprend des
   représentations (pas de reward à prédire), la GA optimise directement la
   fitness (pas de value function). Cohérent avec Pattern #21 (GA > NN pour
   le trading).

## Ce qui reste à prouver (honnêteté)

- **Pas encore de live.** Tout est backtest/rejeu. Le vrai test est FTMO en
  conditions réelles (slippage, latence, gaps, annonces macro).
- **Le champion gen0 (8/8) est peut-être chanceux** — un seul champion sur 40
  validations. Relancer plusieurs seeds pour confirmer la robustesse.
- **Impact du world model entraîné non mesuré** sur le CEM vs modèle
  aléatoire (perte 0.82 mais pas d'ablation sur la planification).
- **Garde-fous (1 %/4 %) jamais testés en stress réel** (gap weekend,
  annonce macro, spike de volatilité).

## Leçon structurante

Les 6 échecs précédents ont établi (Patterns #21/#22) que les NN profonds
échouent sur le bruit M15 et que le GA sur stratégies-règles converge. E.V.A
est le premier projet à intégrer cette leçon dès la conception :
- **Représentations JEPA** au lieu de prédire le reward (évite Pattern #1) ;
- **GA** au lieu de value function (évite Patterns #1/#2) ;
- **Validation holdout systématique** (évite Pattern #15 — promotion aveugle) ;
- **Fitness par généralisation** (évite la dérive en surapprentissage).

## Prochaine étape décisive

Le branchement live MT5 (FTMO/FTUK) est le seul test qui tranchera entre
« méthodologie correcte » et « argent réel ». Prérequis : ISO Windows 11 dans
`~/vms/mt5/iso/` → VM KVM → MT5 + comptes → EA ZMQ → `ConnecteurMT5`.

### Tests de robustesse recommandés avant le live

1. Multi-seeds : relancer l'arène généralisée 5× (graines différentes) et
   vérifier que la proportion de champions généralisant est reproductible.
2. Ablation world model : mesurer l'impact du GRU entraîné sur le CEM vs
   modèle aléatoire (qualité de planification).
3. Stress test garde-fous : simuler gap weekend + spike macro et vérifier
   que le disjoncteur 4 % et le sanitizer 1 % tiennent.
