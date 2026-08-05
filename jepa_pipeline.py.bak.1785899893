"""Pipeline JEPA — préparation et encodage latent sur GPU 0 (PyTorch).

Enchaîne la normalisation multi-horizons/FFT (``DynamicNormalizer``) et
l'encodeur temporel auto-supervisé (``TimeJEPAEncoder`` + ``MomentumTarget``)
pour transformer un flux OHLCV brut en latents 128-dim prêts pour le pont
DLPack vers le moteur JAX.

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
"""

from __future__ import annotations

import torch
from torch import Tensor

from eva import DynamicNormalizer, MomentumTarget, TimeJEPAEncoder

DIM_LATENT: int = 128
DIM_CACHEE: int = 256
NB_TETES: int = 4
NB_COUCHES: int = 3
NB_COEFF_FFT: int = 8


class JEPAPipeline:
    """Encodeur de bout en bout : OHLCV brut -> latents JEPA 128-dim.

    Attributes:
        normalisateur: ``DynamicNormalizer`` (rendements, FFT, RunningLayerNorm).
        modele: ``MomentumTarget`` (encodeur online + cible EMA).
        device: Device PyTorch d'exécution (GPU 0 sur The Hive).
    """

    def __init__(
        self,
        n_canaux_bruts: int = 5,
        dim_latent: int = DIM_LATENT,
        device: str = "cuda:0",
    ) -> None:
        """Initialise le pipeline sur le device demandé.

        Args:
            n_canaux_bruts: Canaux du flux brut (≥ 5 pour OHLCV).
            dim_latent: Dimension de l'espace latent H (128 par spec).
            device: Device PyTorch (``cuda:0`` par défaut).

        Raises:
            RuntimeError: Si le device CUDA demandé est indisponible.
        """
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA indisponible pour device={device}")
        self.device = torch.device(device)
        self.normalisateur = DynamicNormalizer(
            n_canaux_bruts=n_canaux_bruts, nb_coeff_fft=NB_COEFF_FFT
        ).to(self.device)
        encodeur = TimeJEPAEncoder(
            dim_entree=self.normalisateur.dim_sortie,
            dim_latent=dim_latent,
            dim_cachee=DIM_CACHEE,
            nb_tetes=NB_TETES,
            nb_couches=NB_COUCHES,
        ).to(self.device)
        self.modele = MomentumTarget(encodeur).to(self.device)
        self.modele.eval()
        self.normalisateur.eval()

    @torch.no_grad()
    def encoder(self, ohlcv: Tensor) -> Tensor:
        """Encode un flux OHLCV brut en latents 128-dim.

        Args:
            ohlcv: Tenseur brut ``(Batch, Sequence, 5)`` — canaux
                ``[open, high, low, close, volume]``.

        Returns:
            Latents ``(Batch, Sequence, 128)`` sur le device du pipeline.

        Raises:
            ValueError: Si ``ohlcv`` n'est pas 3D.
        """
        if ohlcv.ndim != 3:
            raise ValueError(f"Entrée 3D attendue (B, T, F), reçu {ohlcv.ndim}D")
        features = self.normalisateur(ohlcv.to(self.device))
        _, latents = self.modele(features)
        assert latents.shape[-1] == self.modele.encodeur_online.dim_latent, (
            f"latent {latents.shape[-1]} != "
            f"{self.modele.encodeur_online.dim_latent}"
        )
        return latents

    def pas_entrainement(self, ohlcv: Tensor) -> Tensor:
        """Calcule la perte auto-supervisée JEPA (mode entraînement).

        Args:
            ohlcv: Tenseur brut ``(Batch, Sequence, 5)``.

        Returns:
            Perte scalaire Smooth L1 du bloc futur masqué.
        """
        self.modele.train()
        self.normalisateur.train()
        perte, _ = self.modele(self.normalisateur(ohlcv.to(self.device)))
        self.modele.maj_ema()
        return perte
