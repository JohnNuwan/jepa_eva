"""TimeJEPAEncoder — encodeur temporel auto-supervisé type JEPA léger.

Mini-Transformer 1D (4 têtes, 3 couches, SDPA/FlashAttention-2) projetant une
séquence de features normalisées vers un espace latent H de 128 dimensions,
avec double EMA (MomentumTarget) anti-collapse et perte Smooth-L1 sur la
reconstruction sémantique d'un bloc futur masqué à 20 %.

Conforme au Bloc A.2 de la MASTER-SPECIFICATION E.V.A.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _masque_bloc_futur(
    longueur: int,
    ratio: float,
    generateur: torch.Generator | None,
    appareil: torch.device,
) -> Tensor:
    """Retourne un masque booléen sur un bloc futur contigu.

    Environ ``ratio`` de positions sont masquées, regroupées en un bloc
    contigu ancré dans la seconde moitié (future) de la séquence.

    Args:
        longueur: Longueur de la séquence.
        ratio: Fraction de positions à masquer.
        generateur: Générateur RNG optionnel.
        appareil: Device du tenseur de sortie.

    Returns:
        Masque booléen de forme ``(longueur,)``.
    """
    n_masque = max(1, int(longueur * ratio))
    debut_min = longueur // 2  # bloc forcément dans la seconde moitié
    debut_max = max(debut_min, longueur - n_masque)
    if generateur is not None:
        debut = int(
            torch.randint(debut_min, debut_max + 1, (1,), generator=generateur).item()
        )
    else:
        debut = debut_min
    masque = torch.zeros(longueur, dtype=torch.bool, device=appareil)
    masque[debut : debut + n_masque] = True
    return masque


class CoucheTransformerTemporelle(nn.Module):
    """Bloc Transformer encodeur 1D (pre-norm) avec SDPA forcé."""

    def __init__(self, dim: int, nb_tetes: int, ratio_mlp: int = 4, dropout: float = 0.0) -> None:
        """Initialise la couche Transformer pré-norm.

        Args:
            dim: Dimension du modèle.
            nb_tetes: Nombre de têtes d'attention.
            ratio_mlp: Ratio d'expansion du MLP feed-forward.
            dropout: Taux de dropout sur l'attention.
        """
        super().__init__()
        self.nb_tetes = nb_tetes
        self.dim_tete = dim // nb_tetes
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ratio_mlp),
            nn.GELU(),
            nn.Linear(dim * ratio_mlp, dim),
        )
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        """Applique attention + MLP avec connexions résiduelles.

        Args:
            x: Tenseur d'entrée de forme ``(Batch, Sequence, Dim)``.

        Returns:
            Tenseur de sortie de forme identique.
        """
        b, t, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(b, t, self.nb_tetes, self.dim_tete).transpose(1, 2)
        k = k.view(b, t, self.nb_tetes, self.dim_tete).transpose(1, 2)
        v = v.view(b, t, self.nb_tetes, self.dim_tete).transpose(1, 2)
        # SDPA : sélectionne FlashAttention-2 automatiquement sur Ampere+.
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        o = o.transpose(1, 2).reshape(b, t, d)
        x = x + self.proj(o)
        x = x + self.mlp(self.norm2(x))
        return x


class TimeJEPAEncoder(nn.Module):
    """Encodeur JEPA temporel 1D vers un espace latent H = 128 dimensions.

    Args:
        dim_entree: nombre de features par pas de temps.
        dim_latent: dimension de l'espace latent H (figée à 128 par défaut).
        dim_cachee: dimension cachée du Transformer.
        nb_tetes: têtes d'attention (4 par spec).
        nb_couches: couches Transformer (3 par spec).
    """

    def __init__(
        self,
        dim_entree: int,
        dim_latent: int = 128,
        dim_cachee: int = 256,
        nb_tetes: int = 4,
        nb_couches: int = 3,
    ) -> None:
        """Initialise l'encodeur JEPA temporel.

        Args:
            dim_entree: Nombre de features par pas de temps.
            dim_latent: Dimension de l'espace latent H (128 par spec).
            dim_cachee: Dimension cachée du Transformer.
            nb_tetes: Têtes d'attention (4 par spec).
            nb_couches: Couches Transformer (3 par spec).
        """
        super().__init__()
        self.dim_latent = dim_latent
        self.projection_entree = nn.Linear(dim_entree, dim_cachee)
        self.position = nn.Parameter(torch.zeros(1, 512, dim_cachee))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.couches = nn.ModuleList(
            CoucheTransformerTemporelle(dim_cachee, nb_tetes) for _ in range(nb_couches)
        )
        self.norm_finale = nn.LayerNorm(dim_cachee)
        # Projection finale figée vers l'espace latent H.
        self.tete_latente = nn.Linear(dim_cachee, dim_latent)

    def forward(self, x: Tensor, masque: Tensor | None = None) -> Tensor:
        """Encode une séquence de features vers l'espace latent H.

        Projette chaque pas de temps dans l'espace latent 128-dim via le
        Transformer 1D. Si un masque est fourni, les positions masquées sont
        remplacées par le token moyen du batch avant l'attention.

        Args:
            x: Tenseur de features normalisées de forme
                ``(Batch, Sequence, dim_entree)``.
            masque: Tenseur booléen optionnel de forme ``(Sequence,)`` —
                ``True`` = position neutralisée.

        Returns:
            Tensor de latents de forme ``(Batch, Sequence, dim_latent)``.

        Raises:
            ValueError: Si ``x`` n'est pas 3D, si la dimension des features
                ne correspond pas à ``dim_entree``, ou si la séquence
                dépasse la longueur maximale du position embedding.
        """
        if x.ndim != 3:
            raise ValueError(f"Entrée 3D attendue (B, T, F), reçu {x.ndim}D")
        if x.shape[-1] != self.projection_entree.in_features:
            raise ValueError(
                f"dim_entree={self.projection_entree.in_features} attendue, "
                f"reçu {x.shape[-1]}"
            )
        t = x.shape[1]
        if t > self.position.shape[1]:
            raise ValueError(
                f"Séquence {t} > position embedding {self.position.shape[1]}"
            )
        if masque is not None:
            assert masque.shape == (t,), (
                f"masque {tuple(masque.shape)} != séquence ({t},)"
            )
        h = self.projection_entree(x)
        if masque is not None:
            token_moyen = h.mean(dim=1, keepdim=True)
            h = torch.where(masque.view(1, t, 1), token_moyen.expand_as(h), h)
        h = h + self.position[:, :t]
        for couche in self.couches:
            h = couche(h)
        latents = self.tete_latente(self.norm_finale(h))
        assert latents.shape == (x.shape[0], t, self.dim_latent), (
            f"sortie {tuple(latents.shape)} != "
            f"({x.shape[0]}, {t}, {self.dim_latent})"
        )
        return latents


class MomentumTarget(nn.Module):
    """Double EMA de l'encodeur + prédicteur, mécanisme anti-collapse JEPA.

    Maintient une copie ``requires_grad=False`` de l'encodeur, mise à jour par
    moyenne mobile exponentielle :
        θ_target ← momentum · θ_target + (1 − momentum) · θ_online

    La perte auto-supervisée est une Smooth L1 entre la prédiction du
    prédicteur (depuis le contexte visible) et la cible EMA du bloc futur
    masqué à 20 %, avec stop-gradient sur la cible.
    """

    def __init__(
        self,
        encodeur: TimeJEPAEncoder,
        momentum: float = 0.999,
        ratio_masque: float = 0.20,
    ) -> None:
        """Initialise le double EMA et le prédicteur JEPA.

        Args:
            encodeur: Encodeur online à dupliquer.
            momentum: Coefficient EMA (0.999 par spec).
            ratio_masque: Fraction du bloc futur masqué (0.20 par spec).
        """
        super().__init__()
        self.encodeur_online = encodeur
        self.encodeur_cible = copy.deepcopy(encodeur)
        for p in self.encodeur_cible.parameters():
            p.requires_grad_(False)
        self.momentum = momentum
        self.ratio_masque = ratio_masque
        # Prédicteur léger : contexte visible -> cible future (JEPA).
        self.predicteur = nn.Sequential(
            nn.Linear(encodeur.dim_latent, encodeur.dim_latent * 2),
            nn.GELU(),
            nn.Linear(encodeur.dim_latent * 2, encodeur.dim_latent),
        )

    @torch.no_grad()
    def maj_ema(self) -> None:
        """Mise à jour EMA des poids et buffers de la cible."""
        for p_cible, p_online in zip(
            self.encodeur_cible.parameters(), self.encodeur_online.parameters()
        ):
            p_cible.mul_(self.momentum).add_(p_online.detach(), alpha=1.0 - self.momentum)
        for b_cible, b_online in zip(
            self.encodeur_cible.buffers(), self.encodeur_online.buffers()
        ):
            b_cible.copy_(b_online)

    def forward(
        self,
        x: Tensor,
        generateur: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Calcule la perte auto-supervisée JEPA et les latents du contexte.

        Masque un bloc futur de 20 % de la séquence, encode le contexte
        visible, prédit la cible EMA du bloc masqué, et retourne la perte
        Smooth L1 avec stop-gradient sur la cible.

        Args:
            x: Tenseur de features normalisées de forme
                ``(Batch, Sequence, dim_entree)``.
            generateur: Générateur RNG optionnel pour un masquage
                reproductible.

        Returns:
            Tuple ``(perte, latents_contexte)`` :
                - perte : scalaire Smooth L1 sur le bloc futur masqué,
                - latents_contexte : ``(Batch, Sequence, 128)`` latents
                  online (pour le pont DLPack vers JAX).

        Raises:
            ValueError: Si ``x`` n'est pas un tenseur 3D.
        """
        if x.ndim != 3:
            raise ValueError(f"Entrée 3D attendue (B, T, F), reçu {x.ndim}D")
        t = x.shape[1]
        masque = _masque_bloc_futur(t, self.ratio_masque, generateur, x.device)

        latents_online = self.encodeur_online(x, masque=masque)
        with torch.no_grad():
            latents_cible = self.encodeur_cible(x)  # séquence complète, stop-grad

        prediction = self.predicteur(latents_online)
        m = masque.view(1, t, 1).expand_as(prediction)
        perte = F.smooth_l1_loss(
            prediction[m.bool()].view(-1, self.encodeur_online.dim_latent),
            latents_cible[m.bool()].view(-1, self.encodeur_online.dim_latent),
        )
        return perte, latents_online
