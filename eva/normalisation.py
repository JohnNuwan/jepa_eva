"""DynamicNormalizer — préparation multi-horizons + FFT + normalisation robuste.

Transforme un flux brut OHLCV/ticks en tenseur de features stationnaire :
- rendements glissants sur horizons sémantiques [1, 2, 5, 15, 30, 60] barres,
- filtrage spectral (FFT) des prix pour isoler les composantes cycliques,
- RunningLayerNorm anti-saturation (écrêtage robuste + stats cumulatives).

Conforme au Bloc A.1 de la MASTER-SPECIFICATION E.V.A.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

HORIZONS: tuple[int, ...] = (1, 2, 5, 15, 30, 60)


class RunningLayerNorm(nn.Module):
    """Normalisation par couche avec statistiques cumulatives (moyenne/variance).

    À chaque forward en mode ``train``, la moyenne et la variance sont mises à
    jour par moyenne mobile cumulée, puis les valeurs extrêmes (annonces macro,
    spikes de volatilité) sont écrêtées par clipping robuste avant projection
    dans une distribution standard N(0, 1).

    Args:
        dim: dimension des features normalisées.
        momentum: poids de l'échantillon courant dans la mise à jour cumulative.
        clip_std: seuil d'écrêtage en nombre d'écarts-types (winsorisation).
        eps: stabilisateur numérique.
    """

    def __init__(
        self,
        dim: int,
        momentum: float = 0.01,
        clip_std: float = 5.0,
        eps: float = 1e-5,
    ) -> None:
        """Initialise la RunningLayerNorm.

        Args:
            dim: Dimension des features normalisées.
            momentum: Poids de l'échantillon courant dans la mise à jour.
            clip_std: Seuil d'écrêtage en écarts-types.
            eps: Stabilisateur numérique.
        """
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.clip_std = clip_std
        self.eps = eps

        self.register_buffer("compteur", torch.zeros((), dtype=torch.long))
        self.register_buffer("moyenne_courante", torch.zeros(dim))
        self.register_buffer("variance_courante", torch.ones(dim))
        # Paramètres affines appris (LayerNorm classique).
        self.poids = nn.Parameter(torch.ones(dim))
        self.biais = nn.Parameter(torch.zeros(dim))

    def _maj_stats(self, x: Tensor) -> None:
        """Met à jour moyenne/variance cumulatives (sans gradient)."""
        with torch.no_grad():
            x_plat = x.detach().reshape(-1, x.shape[-1]).float()
            moy_batch = x_plat.mean(dim=0)
            var_batch = x_plat.var(dim=0, unbiased=False)
            m = self.momentum
            self.moyenne_courante.mul_(1.0 - m).add_(moy_batch * m)
            # Variance combinée (terme inter-batch inclus, correction de groupe).
            delta = moy_batch - self.moyenne_courante
            self.variance_courante.mul_(1.0 - m).add_(
                (var_batch + delta.pow(2) * m) * m
            )
            self.compteur += 1

    def forward(self, x: Tensor) -> Tensor:
        """Normalise ``x`` de forme (..., dim) vers N(0, 1) écrêtée."""
        if self.training:
            self._maj_stats(x)

        moyenne = self.moyenne_courante.to(dtype=x.dtype, device=x.device)
        ecart = (
            self.variance_courante.to(dtype=x.dtype, device=x.device)
            .add(self.eps)
            .sqrt()
        )
        z = (x - moyenne) / ecart
        # Anti-saturation : winsorisation des outliers macro à ±clip_std σ.
        z = torch.clamp(z, -self.clip_std, self.clip_std)
        return z * self.poids + self.biais


class DynamicNormalizer(nn.Module):
    """Pipeline de préparation des données brutes OHLCV/ticks.

    Entrée : tenseur ``(B, T, F)`` où les 5 premiers canaux sont
    ``[open, high, low, close, volume]`` (les suivants, ex. spread, sont
    normalisés directement).

    Sortie : tenseur ``(B, T, F')`` stationnaire, prêt pour l'encodeur JEPA.
    """

    def __init__(
        self,
        n_canaux_bruts: int,
        nb_coeff_fft: int = 8,
        horizons: tuple[int, ...] = HORIZONS,
        clip_std: float = 5.0,
        eps: float = 1e-8,
    ) -> None:
        """Initialise le pipeline de normalisation.

        Args:
            n_canaux_bruts: Nombre de canaux du flux brut (≥ 5 pour OHLCV).
            nb_coeff_fft: Nombre de composantes spectrales conservées.
            horizons: Horizons de rendement en barres.
            clip_std: Seuil d'écrêtage de la RunningLayerNorm.
            eps: Stabilisateur numérique.

        Raises:
            ValueError: Si ``n_canaux_bruts`` < 5.
        """
        super().__init__()
        if n_canaux_bruts < 5:
            raise ValueError("Au moins 5 canaux OHLCV requis.")
        self.n_canaux_bruts = n_canaux_bruts
        self.nb_coeff_fft = nb_coeff_fft
        self.horizons = horizons
        self.eps = eps

        # Dimension de sortie :
        #   - 1 log-prix normalisé,
        #   - 4 ranges intrabarre normalisés (O/H/L/C relatifs),
        #   - len(horizons) rendements multi-horizons,
        #   - 2 * nb_coeff_fft coefficients spectraux (réel/imag),
        #   - (n_canaux_bruts - 5) canaux supplémentaires normalisés.
        self.dim_sortie = (
            1 + 4 + len(horizons) + 2 * nb_coeff_fft + (n_canaux_bruts - 5)
        )
        self.norm = RunningLayerNorm(self.dim_sortie, clip_std=clip_std)

    def _rendements(self, close: Tensor) -> Tensor:
        """Rendements logarithmiques multi-horizons sur le close.

        Args:
            close: ``(B, T)`` prix de clôture.

        Returns:
            ``(B, T, len(horizons))`` rendements log, zéro-padding en tête.
        """
        log_close = torch.log(close.clamp_min(self.eps))
        sorties = []
        for h in self.horizons:
            r = log_close[:, h:] - log_close[:, :-h]
            # Pad temporel (dimension 1) en tête : (gauche, droite) sur la dim -2.
            r = torch.nn.functional.pad(r.unsqueeze(-1), (0, 0, h, 0)).squeeze(-1)
            sorties.append(r)
        return torch.stack(sorties, dim=-1)

    def _fft_glissante(self, close: Tensor) -> Tensor:
        """Composantes spectrales du close via FFT sur la fenêtre entière.

        Isole les ``nb_coeff_fft`` premières composantes cycliques (hors DC)
        et retourne parties réelle/imaginaire normalisées.

        Args:
            close: ``(B, T)`` prix de clôture.

        Returns:
            ``(B, T, 2 * nb_coeff_fft)`` coefficients interpolés sur T.
        """
        x = close - close.mean(dim=1, keepdim=True)
        spectre = torch.fft.rfft(x, dim=1)  # (B, T//2+1) complexe
        k = min(self.nb_coeff_fft, spectre.shape[1] - 1)
        # On saute la composante DC (indice 0) : cycles purs.
        coeff = spectre[:, 1 : k + 1]
        reel = coeff.real / close.shape[1]
        imag = coeff.imag / close.shape[1]
        # Diffusion temporelle : chaque barre reçoit les mêmes coefficients
        # (caractéristique cyclique de la fenêtre complète).
        reel = reel.unsqueeze(1).expand(-1, close.shape[1], -1)
        imag = imag.unsqueeze(1).expand(-1, close.shape[1], -1)
        if k < self.nb_coeff_fft:  # séquence courte : zéro-padding
            pad = (0, self.nb_coeff_fft - k)
            reel = torch.nn.functional.pad(reel, pad)
            imag = torch.nn.functional.pad(imag, pad)
        return torch.cat([reel, imag], dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Transforme le flux brut OHLCV en features normalisées.

        Calcule les rendements multi-horizons, les ranges intrabarre, les
        composantes spectrales FFT, puis applique la ``RunningLayerNorm``
        anti-saturation. La sortie est stationnaire et prête pour JEPA.

        Args:
            x: Tenseur brut de forme ``(Batch, Sequence, n_canaux_bruts)``,
                canaux 0..4 = ``[open, high, low, close, volume]``.

        Returns:
            Tensor de features de forme ``(Batch, Sequence, dim_sortie)``.

        Raises:
            ValueError: Si ``x`` n'est pas 3D ou si le nombre de canaux ne
                correspond pas à ``n_canaux_bruts``.
        """
        if x.ndim != 3:
            raise ValueError(f"Entrée 3D attendue (B, T, F), reçu {x.ndim}D")
        if x.shape[-1] != self.n_canaux_bruts:
            raise ValueError(
                f"{self.n_canaux_bruts} canaux attendus, reçu {x.shape[-1]}"
            )
        o, h, bas, c, v = (x[..., i] for i in range(5))

        log_c = torch.log(c.clamp_min(self.eps))
        log_c = log_c - log_c.mean(dim=1, keepdim=True)

        # Ranges intrabarre relatifs (invariants d'échelle).
        ref = c.clamp_min(self.eps)
        ranges = torch.stack(
            [
                (o - c) / ref,
                (h - c) / ref,
                (bas - c) / ref,
                (h - bas) / ref,
            ],
            dim=-1,
        )

        rendements = self._rendements(c)
        spectral = self._fft_glissante(c)

        # Volume : log-variation robuste.
        log_v = torch.log(v.clamp_min(self.eps))
        log_v = log_v - log_v.mean(dim=1, keepdim=True)
        log_v = log_v.unsqueeze(-1)

        features = [log_c.unsqueeze(-1), ranges, rendements, spectral]
        if self.n_canaux_bruts > 5:  # canaux exogènes (spread, ticks…)
            features.append(x[..., 5:])
        else:
            features.append(log_v)
        sortie = torch.cat(features, dim=-1)
        # Ajustement dimensionnel exact (volume vs canaux exogènes).
        if sortie.shape[-1] > self.dim_sortie:
            sortie = sortie[..., : self.dim_sortie]
        elif sortie.shape[-1] < self.dim_sortie:
            sortie = torch.nn.functional.pad(
                sortie, (0, self.dim_sortie - sortie.shape[-1])
            )
        return self.norm(sortie)
