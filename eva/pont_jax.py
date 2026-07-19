"""JAXTransitionBridge — pont mémoire zéro-copie PyTorch -> JAX via DLPack.

Réceptionne le tenseur latent 128-dim produit par l'encodeur PyTorch (GPU 0)
et le convertit en tableau JAX résidant sur le GPU 1, sans recopie mémoire
VRAM quand le device cible est identique, sinon via ``jax.device_put``
optimisé par XLA.

Conforme au Bloc B.1 de la MASTER-SPECIFICATION E.V.A.
"""

from __future__ import annotations

import jax
import torch
from jax import Array


class JAXTransitionBridge:
    """Convertisseur DLPack PyTorch -> JAX avec placement GPU contrôlé.

    Args:
        device_jax_cible: device JAX de destination (ex. ``jax.devices()[1]``
            pour le GPU 1 dédié au core JAX). Si ``None``, le device par
            défaut JAX est utilisé.
    """

    def __init__(self, device_jax_cible: "jax.Device | None" = None) -> None:
        """Initialise le pont DLPack.

        Args:
            device_jax_cible: Device JAX de destination (ex. GPU 1). Si
                ``None``, le device par défaut JAX est utilisé.
        """
        self.device_cible = device_jax_cible

    def convertir(self, tenseur_torch: torch.Tensor) -> Array:
        """Convertit un tenseur PyTorch en tableau JAX via DLPack.

        Le passage par ``jax.dlpack.from_dlpack`` garantit une latence de
        transfert nulle (partage de la même allocation VRAM) lorsque JAX et
        PyTorch résident sur le même GPU ; sinon XLA effectue un transfert
        pair-à-pair optimisé vers ``device_cible``.

        Args:
            tenseur_torch: Tenseur PyTorch (CPU ou CUDA) de forme
                ``(Batch, 128)`` ou ``(128,)``, convertible en float32.

        Returns:
            Tableau JAX de forme identique, placé sur ``device_cible``
            (ou device par défaut si non spécifié).

        Raises:
            TypeError: Si l'entrée n'est pas un ``torch.Tensor``.
            ValueError: Si le tenseur est vide ou a plus de 3 dimensions
                (forme inattendue pour un latent JEPA).
        """
        if not isinstance(tenseur_torch, torch.Tensor):
            raise TypeError(
                f"torch.Tensor attendu, reçu {type(tenseur_torch).__name__}"
            )
        if tenseur_torch.ndim == 0 or tenseur_torch.ndim > 3:
            raise ValueError(
                f"Tenseur 1D/2D/3D attendu, reçu {tenseur_torch.ndim}D "
                f"de forme {tuple(tenseur_torch.shape)}"
            )
        if tenseur_torch.numel() == 0:
            raise ValueError("Tenseur vide — conversion refusée")
        tenseur = tenseur_torch.detach().contiguous()
        if tenseur.dtype != torch.float32:
            tenseur = tenseur.float()
        tableau = jax.dlpack.from_dlpack(tenseur)
        if self.device_cible is not None:
            tableau = jax.device_put(tableau, self.device_cible)
        assert tableau.shape == tuple(tenseur.shape), (
            f"forme JAX {tableau.shape} != forme torch {tuple(tenseur.shape)}"
        )
        return tableau

    def convertir_batch(self, tenseurs: list[torch.Tensor]) -> Array:
        """Empile et convertit une liste de tenseurs PyTorch en un batch JAX.

        Args:
            tenseurs: liste de ``torch.Tensor`` de formes identiques.

        Returns:
            Tableau JAX de forme ``(len(tenseurs), *forme)``.
        """
        empile = torch.stack([t.detach().contiguous() for t in tenseurs])
        return self.convertir(empile)


def pont_defaut() -> JAXTransitionBridge:
    """Fabrique un pont ciblant le dernier GPU JAX disponible (GPU 1 si dual).

    Returns:
        ``JAXTransitionBridge`` configuré pour le GPU dédié au core JAX.
    """
    gpus = [d for d in jax.devices() if d.platform == "gpu"]
    cible = gpus[-1] if gpus else None
    return JAXTransitionBridge(device_jax_cible=cible)
