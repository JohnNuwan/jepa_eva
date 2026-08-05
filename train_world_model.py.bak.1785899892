"""Entraînement supervisé du World Model GRU (transitions latentes réelles).

Entraîne le GRU compact à prédire la transition H_t × A_t → H_{t+1} sur les
latents JEPA réels : le modèle apprend la dynamique de l'espace latent, ce
qui permet au planificateur CEM de simuler des trajectoires plausibles au
lieu d'un modèle aléatoire. Perte = MSE sur le latent suivant + récompense.

Usage :
    PYTHONPATH=. venv/bin/python train_world_model.py --symbole XAUUSD --steps 500

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_arena import (
    DIM_ACTION,
    DIM_CACHE,
    DIM_LATENT,
    ParametresWorldModel,
    appliquer_gru,
    initialiser_world_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.train_wm")


def parse_args() -> argparse.Namespace:
    """Analyse les arguments CLI.

    Returns:
        Espace de noms avec symbole, timeframe, steps et hyperparamètres.
    """
    p = argparse.ArgumentParser(description="Entraînement World Model GRU")
    p.add_argument("--symbole", default="XAUUSD")
    p.add_argument("--timeframe", default="m15")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--frac_entrainement", type=float, default=0.8)
    p.add_argument("--sortie", default="checkpoints_wm")
    return p.parse_args()


def descente_adam(
    params: ParametresWorldModel,
    grads: ParametresWorldModel,
    etat_m: ParametresWorldModel,
    etat_v: ParametresWorldModel,
    lr: float,
    t: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[ParametresWorldModel, ParametresWorldModel, ParametresWorldModel]:
    """Une étape d'optimiseur Adam (pur JAX, sans optax).

    Args:
        params: Poids courants.
        grads: Gradients de la perte.
        etat_m: Premier moment.
        etat_v: Second moment.
        lr: Taux d'apprentissage.
        t: Itération (pour correction de biais).
        beta1, beta2, eps: Hyperparamètres Adam.

    Returns:
        Tuple ``(nouveaux_params, nouvel_etat_m, nouvel_etat_v)``.
    """
    m = jax.tree.map(lambda m_, g: beta1 * m_ + (1 - beta1) * g, etat_m, grads)
    v = jax.tree.map(lambda v_, g: beta2 * v_ + (1 - beta2) * g**2, etat_v, grads)
    m_hat = jax.tree.map(lambda m_: m_ / (1 - beta1**t), m)
    v_hat = jax.tree.map(lambda v_: v_ / (1 - beta2**t), v)
    nouveaux = jax.tree.map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
    )
    return nouveaux, m, v


def entrainer(args: argparse.Namespace) -> None:
    """Boucle d'entraînement supervisé du world model.

    Args:
        args: Arguments CLI.

    Raises:
        FileNotFoundError: Si les latents sont absents.
    """
    chemin_latents = Path("latents") / f"{args.symbole}_{args.timeframe}_latents.npz"
    if not chemin_latents.is_file():
        raise FileNotFoundError(f"Latents absents : {chemin_latents}")
    donnees = np.load(chemin_latents)
    latents = jnp.asarray(donnees["latents"], dtype=jnp.float32)
    prix = jnp.asarray(donnees["prix"], dtype=jnp.float32)
    nb = int(latents.shape[0])
    limite_train = int(nb * args.frac_entrainement)

    # Cible de récompense : variation relative du prix à t+1 (mark-to-market).
    rendements = (prix[1:] - prix[:-1]) / jnp.maximum(prix[:-1], 1e-8)

    cle = jax.random.PRNGKey(0)
    params = initialiser_world_model(cle, DIM_LATENT, DIM_ACTION, DIM_CACHE)
    zeros = jax.tree.map(jnp.zeros_like, params)
    etat_m, etat_v = zeros, zeros

    def perte_fn(
        p: ParametresWorldModel,
        h_t: jnp.ndarray,
        action: jnp.ndarray,
        h_next: jnp.ndarray,
        recompense_cible: jnp.ndarray,
        cache0: jnp.ndarray,
    ) -> jnp.ndarray:
        """MSE sur latent suivant prédit + récompense prédite."""
        def un_pas(c: jnp.ndarray, entree: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
            h, a = entree
            nc, r, _ = appliquer_gru(p, h, a, c)
            return nc, (nc, r)

        _, (h_pred, r_pred) = jax.lax.scan(
            un_pas, cache0, (h_t, action)
        )
        # MSE latents (projection du cache vers l'espace latent via troncature)
        mse_latent = jnp.mean((h_pred[:, : DIM_LATENT] - h_next) ** 2)
        mse_recompense = jnp.mean((r_pred - recompense_cible) ** 2)
        return mse_latent + 0.5 * mse_recompense

    grad_fn = jax.jit(jax.value_and_grad(perte_fn))

    journal.info(
        "Début : %d transitions | %d steps | batch=%d | lr=%.1e",
        limite_train, args.steps, args.batch, args.lr,
    )
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    pertes: list[float] = []

    for step in range(1, args.steps + 1):
        idx = rng.integers(1, limite_train - 2, size=args.batch)
        h_t = latents[idx]
        h_next = latents[idx + 1]
        # Action factice : direction = signe du rendement réel (signal causal).
        rend = rendements[idx]
        action = jnp.concatenate(
            [
                jnp.tanh(rend * 500.0)[:, None],
                jnp.abs(jnp.tanh(rend * 500.0))[:, None],
                jnp.zeros((args.batch, DIM_ACTION - 2)),
            ],
            axis=1,
        )
        cache0 = jnp.zeros(DIM_CACHE)
        perte, grads = grad_fn(
            params, h_t, action, h_next, rend, cache0
        )
        params, etat_m, etat_v = descente_adam(
            params, grads, etat_m, etat_v, args.lr, step
        )
        pertes.append(float(perte))
        if step % 50 == 0:
            journal.info(
                "step %4d/%d | perte(50)=%.6f | %.0f step/s",
                step, args.steps, float(np.mean(pertes[-50:])),
                step / max(1e-9, time.perf_counter() - t0),
            )

    dossier = Path(args.sortie)
    dossier.mkdir(parents=True, exist_ok=True)
    aplati, _ = jax.tree.flatten(params)
    donnees_poids = {f"p{i}": np.asarray(f) for i, f in enumerate(aplati)}
    chemin = dossier / f"world_model_{args.symbole}_{args.timeframe}.npz"
    np.savez_compressed(chemin, **donnees_poids)
    moy_debut = float(np.mean(pertes[:50]))
    moy_fin = float(np.mean(pertes[-50:]))
    journal.info(
        "Terminé : perte %.6f -> %.6f | world model -> %s", moy_debut, moy_fin, chemin
    )


def main() -> None:
    """Point d'entrée CLI."""
    entrainer(parse_args())


if __name__ == "__main__":
    main()
