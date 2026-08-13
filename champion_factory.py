#!/usr/bin/env python3
"""Champion Factory E.V.A — génération massive de champions rentables.

Fabrique paramétrable qui pilote l'arène génétique JAX (``jax_arena.py``)
avec des VARIANTES DE STRATÉGIE :

- fitness configurable (poids Sortino / Sharpe / drawdown / net profit /
  profit factor / win rate) via patch de ``jax_arena._calculer_fitness``,
- coût de transaction variable (sélectivité des trades),
- capacité GRU variable (dim_cache),
- pression d'évolution variable (population, élites, mutation),
- régime d'entraînement variable (segment, generations, frac_train),
- validation holdout ou walk-forward (anti-overfitting),
- seed variable (diversité génétique).

Chaque run produit :
- registry_massive/<run_id>/registry.jsonl          (champions validés)
- registry_massive/<run_id>/candidates.jsonl        (tous les records)
- registry_massive/<run_id>/champions/*.npz         (poids Pytree)
- registry_massive/<run_id>/run_meta.json           (config + métriques)

Usage :
    PYTHONPATH=. XLA_PYTHON_CLIENT_MEM_FRACTION=0.12 .venv/bin/python3 \
        champion_factory.py --config configs/US30_baseline.json

PEP 8 / PEP 484 / docstrings Google.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jax_arena as ja

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.factory")

# Seuils de validation holdout (défauts, surchargés par config).
DD_MAX_HOLDOUT: float = 5.0
NP_MIN_HOLDOUT: float = 0.0


# ---------------------------------------------------------------------------
# Fitness paramétrable (patch de jax_arena._calculer_fitness)
# ---------------------------------------------------------------------------
_FITNESS_CFG: dict = {"sortino": 2.0, "sharpe": 0.0, "dd": 1.0, "np": 1.0,
                      "pf": 0.0, "wr": 0.0}


def _calculer_fitness_variant(
    etat_final: ja.EtatSimulation,
    capital_initial: float,
) -> ja.ResultatEvaluation:
    """Fitness paramétrable : Σ poids × métriques − DD + NP.

    Args:
        etat_final: État terminal d'un agent après simulation.
        capital_initial: Capital de départ.

    Returns:
        ``ResultatEvaluation`` avec fitness = w_sortino·Sortino
        + w_sharpe·Sharpe − w_dd·DD + w_np·NP + w_pf·PF + w_wr·WR.
    """
    retours = etat_final.historique_retours
    moyenne = jnp.mean(retours)
    retours_neg = jnp.minimum(retours, 0.0)
    downside_std = jnp.sqrt(jnp.mean(retours_neg**2) + 1e-12)
    sortino = moyenne / downside_std
    vol = jnp.std(retours) + 1e-12
    sharpe = moyenne / vol
    drawdown = (etat_final.equity_peak - etat_final.cash) / jnp.maximum(
        etat_final.equity_peak, 1e-8
    )
    max_dd_pct = drawdown * 100.0
    net_profit = (etat_final.cash - capital_initial) / capital_initial * 100.0
    nb_trades = etat_final.nb_trades
    win_rate = jnp.where(
        nb_trades > 0.0,
        etat_final.trades_gagnants / jnp.maximum(nb_trades, 1.0) * 100.0,
        0.0,
    )
    profit_factor = etat_final.profit_brut / jnp.maximum(
        etat_final.perte_brute, 1e-9
    )
    cfg = _FITNESS_CFG
    fitness = (
        cfg["sortino"] * sortino
        + cfg["sharpe"] * sharpe
        - cfg["dd"] * max_dd_pct
        + cfg["np"] * net_profit
        + cfg["pf"] * profit_factor
        + cfg["wr"] * win_rate * 0.01
    )
    return ja.ResultatEvaluation(
        fitness=fitness,
        net_profit=net_profit,
        max_drawdown=max_dd_pct,
        sortino=sortino,
        win_rate=win_rate,
        profit_factor=profit_factor,
        nb_trades=nb_trades,
    )


def _appliquer_config_arene(cfg: dict) -> None:
    """Applique les paramètres globaux de l'arène (patch module).

    Args:
        cfg: Config du run (cout_transaction, fitness).
    """
    ja.COUT_TRANSACTION = float(cfg.get("cout_transaction", 0.0002))
    ja.LEVIER = float(cfg.get("levier", 1.0))
    _FITNESS_CFG.update(cfg.get("fitness", {}))
    ja._calculer_fitness = _calculer_fitness_variant
    # Injection de la config dans le module pour traçabilité.
    ja.FACTORY_CFG = cfg


# ---------------------------------------------------------------------------
# Évaluation holdout / walk-forward
# ---------------------------------------------------------------------------
def evaluer_agent(
    arene: ja.JaxGeneticArena,
    params: object,
    prix_h: jnp.ndarray,
    latents_h: jnp.ndarray,
) -> dict[str, float]:
    """Évalue un champion unique sur un segment holdout.

    Args:
        arene: Arène (pour la jit d'évaluation).
        params: Pytree du champion (sans dimension population).
        prix_h: Prix holdout ``(nb_pas,)``.
        latents_h: Latents holdout ``(nb_pas, 128)``.

    Returns:
        Métriques holdout.
    """
    arene_1 = ja.JaxGeneticArena(
        jax.random.PRNGKey(0), taille_population=1,
        dim_cache=arene.dim_cache, dim_action=arene.dim_action,
    )
    arene_1.population = jax.tree.map(lambda p: p[None], params)
    res = arene_1.evaluer_population(prix_h, latents_h)
    return {
        "fitness": float(res.fitness[0]),
        "net_profit": float(res.net_profit[0]),
        "drawdown": float(res.max_drawdown[0]),
        "sortino": float(res.sortino[0]),
        "win_rate": float(res.win_rate[0]),
        "profit_factor": float(res.profit_factor[0]),
        "nb_trades": float(res.nb_trades[0]),
    }


def generalise(m_h: dict[str, float], cfg: dict) -> bool:
    """Vérifie la généralisation holdout.

    Args:
        m_h: Métriques holdout.
        cfg: Config (seuils surchargés).

    Returns:
        ``True`` si rentable et drawdown maîtrisé.
    """
    return (
        m_h["net_profit"] > float(cfg.get("np_min_holdout", NP_MIN_HOLDOUT))
        and m_h["drawdown"] <= float(cfg.get("dd_max_holdout", DD_MAX_HOLDOUT))
    )


def sauvegarder_poids(
    population: ja.ParametresWorldModel,
    idx: int,
    dossier: Path,
    tag: str,
    metriques: dict[str, float],
) -> Path:
    """Sauvegarde les poids d'un champion en ``.npz``.

    Args:
        population: Pytree population.
        idx: Indice du champion.
        dossier: Dossier de sortie.
        tag: Étiquette du fichier.
        metriques: Métriques embarquées.

    Returns:
        Chemin du fichier.
    """
    dossier.mkdir(parents=True, exist_ok=True)
    poids = jax.tree.map(lambda p: np.asarray(p[idx]), population)
    aplati, _ = jax.tree.flatten(poids)
    donnees = {f"p{i}": np.asarray(f) for i, f in enumerate(aplati)}
    for k, v in metriques.items():
        donnees[k] = np.asarray(float(v))
    chemin = dossier / f"{tag}.npz"
    np.savez_compressed(chemin, **donnees)
    return chemin


# ---------------------------------------------------------------------------
# Run principal
# ---------------------------------------------------------------------------
def lancer_run(cfg: dict) -> dict:
    """Exécute un run complet de fabrication de champions.

    Args:
        cfg: Config du run.

    Returns:
        Résumé des résultats (champions, records, durée).
    """
    t0 = time.perf_counter()
    run_id = cfg["id"]
    sym = cfg["symbole"]
    tf = cfg.get("timeframe", "m15")
    base = Path(cfg.get("base_dir", "registry_massive"))
    dossier_run = base / run_id
    dossier_champions = dossier_run / "champions"
    dossier_run.mkdir(parents=True, exist_ok=True)

    registry_path = dossier_run / "registry.jsonl"
    candidates_path = dossier_run / "candidates.jsonl"

    # --- Données ----------------------------------------------------------
    chemin_latents = Path("latents") / f"{sym}_{tf}_latents.npz"
    if not chemin_latents.is_file():
        raise FileNotFoundError(f"Latents absents : {chemin_latents}")
    donnees = np.load(chemin_latents)
    prix = jnp.asarray(donnees["prix"], dtype=jnp.float32)
    latents = jnp.asarray(donnees["latents"], dtype=jnp.float32)
    nb = int(prix.shape[0])

    walkforward = int(cfg.get("walkforward", 1))
    frac_train = float(cfg.get("frac_train", 0.8))
    nb_folds = walkforward
    taille_fold = nb // nb_folds
    seuil_holdout = int(nb * frac_train) if nb_folds == 1 else None

    # --- Arène ------------------------------------------------------------
    _appliquer_config_arene(cfg)
    cle = jax.random.PRNGKey(int(cfg.get("seed", 0)))
    arene = ja.JaxGeneticArena(
        cle,
        taille_population=int(cfg.get("taille_pop", 64)),
        dim_cache=int(cfg.get("dim_cache", 256)),
        dim_action=int(cfg.get("dim_action", 8)),
    )

    generations = int(cfg.get("generations", 200))
    segment = int(cfg.get("segment", 512))
    nb_elites = int(cfg.get("nb_elites", 16))
    taux_mutation = float(cfg.get("taux_mutation", 0.1))
    sigma_mutation = float(cfg.get("sigma_mutation", 0.02))
    eval_holdout = int(cfg.get("eval_holdout", 10))

    nb_champions = 0
    nb_records = 0
    meilleur_holdout = -np.inf

    journal.info(
        "RUN %s | %s | pop=%d dim_cache=%d cost=%.5f seg=%d gens=%d wf=%d",
        run_id, sym, int(cfg.get("taille_pop", 64)),
        int(cfg.get("dim_cache", 256)),
        float(cfg.get("cout_transaction", 0.0002)), segment,
        generations, nb_folds,
    )

    for fold in range(nb_folds):
        if nb_folds == 1:
            limite_train = seuil_holdout
            idx_h = slice(limite_train, nb)
        else:
            idx_h = slice(fold * taille_fold, (fold + 1) * taille_fold)
            limite_train = max(0, fold * taille_fold)
            if fold == nb_folds - 1:
                limite_train = nb - taille_fold

        prix_h = prix[idx_h]
        latents_h = latents[idx_h]
        # Limiter la taille du holdout évalué (coût GPU).
        max_holdout = int(cfg.get("max_holdout", 1500))
        if prix_h.shape[0] > max_holdout:
            prix_h = prix_h[-max_holdout:]
            latents_h = latents_h[-max_holdout:]

        record_fold = -np.inf

        for gen in range(generations):
            debut_max = max(1, limite_train - segment)
            if debut_max <= 0:
                debut_max = 1
            debut = int(jax.random.randint(
                jax.random.PRNGKey(fold * 100000 + gen), (), 0, debut_max
            ))
            prix_seg = jax.lax.dynamic_slice_in_dim(prix, debut, segment)
            latents_seg = jax.lax.dynamic_slice_in_dim(latents, debut, segment)

            res = arene.evaluer_population(prix_seg, latents_seg)
            idx_best = int(jnp.argmax(res.fitness))
            fitness_best = float(res.fitness[idx_best])
            record_fold = max(record_fold, fitness_best)

            # Candidate record (tous les records, traçabilité).
            if fitness_best > record_fold - 1e-9 and gen % eval_holdout == 0:
                m_rec = {
                    "fitness": float(res.fitness[idx_best]),
                    "net_profit": float(res.net_profit[idx_best]),
                    "drawdown": float(res.max_drawdown[idx_best]),
                    "sortino": float(res.sortino[idx_best]),
                    "win_rate": float(res.win_rate[idx_best]),
                    "profit_factor": float(res.profit_factor[idx_best]),
                    "nb_trades": float(res.nb_trades[idx_best]),
                }
                entree = {
                    "run_id": run_id, "fold": fold, "generation": gen,
                    **{k: round(float(v), 4) for k, v in m_rec.items()},
                }
                with candidates_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entree, ensure_ascii=False) + "\n")
                nb_records += 1

            # Validation holdout périodique.
            if (gen + 1) % eval_holdout == 0:
                params_best = jax.tree.map(lambda p: p[idx_best], arene.population)
                m_h = evaluer_agent(arene, params_best, prix_h, latents_h)
                if generalise(m_h, cfg) and m_h["fitness"] > meilleur_holdout:
                    meilleur_holdout = m_h["fitness"]
                    nb_champions += 1
                    chemin = sauvegarder_poids(
                        arene.population, idx_best, dossier_champions,
                        f"champion_f{fold}_g{gen}", m_h,
                    )
                    entree = {
                        "run_id": run_id, "fold": fold, "generation": gen,
                        "fitness_train": round(fitness_best, 4),
                        **{k: round(float(v), 4) for k, v in m_h.items()},
                        "fichier": chemin.name,
                    }
                    with registry_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(entree, ensure_ascii=False) + "\n")
                    journal.info(
                        "  ✓ fold %d gen %d : GÉNÉRALISE np=%+.2f%% dd=%.2f%% "
                        "wr=%.1f%% pf=%.2f -> %s",
                        fold, gen, m_h["net_profit"], m_h["drawdown"],
                        m_h["win_rate"], m_h["profit_factor"], chemin.name,
                    )
                else:
                    journal.info(
                        "  ✗ fold %d gen %d : holdout np=%+.2f%% dd=%.2f%%",
                        fold, gen, m_h["net_profit"], m_h["drawdown"],
                    )

            arene.evoluer(
                res.fitness, jax.random.PRNGKey(fold * 100000 + 1000 + gen),
                nb_elites=nb_elites, taux_mutation=taux_mutation,
                sigma_mutation=sigma_mutation,
            )

            if (gen + 1) % 50 == 0:
                journal.info(
                    "run %s fold %d gen %3d/%d | best=%.3f | champions=%d | %.1f gen/s",
                    run_id, fold, gen + 1, generations, fitness_best,
                    nb_champions, (gen + 1) / max(1e-9, time.perf_counter() - t0),
                )

    # --- Sauvegarde du champion final (meilleur de la dernière génération) ---
    res_final = arene.evaluer_population(prix[-segment:], latents[-segment:])
    idx_final = int(jnp.argmax(res_final.fitness))
    params_final = jax.tree.map(lambda p: p[idx_final], arene.population)
    m_final = evaluer_agent(arene, params_final, prix_h, latents_h)
    chemin_final = sauvegarder_poids(
        arene.population, idx_final, dossier_champions, "champion_final", m_final,
    )
    entree_final = {
        "run_id": run_id, "fold": "final", "generation": generations,
        "fitness_train": float(res_final.fitness[idx_final]),
        **{k: round(float(v), 4) for k, v in m_final.items()},
        "fichier": chemin_final.name,
    }
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entree_final, ensure_ascii=False) + "\n")
    nb_champions += 1

    duree = time.perf_counter() - t0
    resume = {
        "run_id": run_id, "symbole": sym, "duree_s": round(duree, 1),
        "nb_champions": nb_champions, "nb_records": nb_records,
        "meilleur_holdout": round(float(meilleur_holdout), 4),
        "config": cfg,
    }
    with (dossier_run / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(resume, f, ensure_ascii=False, indent=2)

    journal.info(
        "RUN %s TERMINÉ : %d champions | record_holdout=%.3f | %.0fs",
        run_id, nb_champions, meilleur_holdout, duree,
    )
    return resume


def main() -> None:
    """Point d'entrée CLI."""
    p = argparse.ArgumentParser(description="Champion Factory E.V.A")
    p.add_argument("--config", required=True, help="Fichier JSON de config")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    resume = lancer_run(cfg)
    print(json.dumps(resume, ensure_ascii=False))


if __name__ == "__main__":
    main()
