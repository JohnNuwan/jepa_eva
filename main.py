"""Orchestrateur maître E.V.A — boucle de trading infinie sur The Hive.

Chaîne complète à chaque tick :
    flux OHLCV -> ``jepa_pipeline.py`` (latents 128-dim, GPU 0)
    -> pont DLPack zéro-copie (``jax_arena.bridge_pytorch_to_jax``)
    -> planification CEM 5 000 trajectoires (``jax_arena.TDMPC2Planner``, GPU 1)
    -> filtre déterministe 1 % (``action_sanitizer.ActionSanitizer``)
    -> disjoncteur dur 4 % (``action_sanitizer.DrawdownDisconnector``)
    -> émission de l'ordre validé.

Garde-fous : le disjoncteur est vérifié AVANT toute émission ; toute
exception ciblée est journalisée sans interrompre la boucle (production).

Conforme PEP 8 / PEP 484 / PEP 257 (docstrings Google en français).
Exécution : PYTHONPATH=. venv/bin/python main.py
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from dataclasses import dataclass

import jax
import numpy as np
import torch

from action_sanitizer import (
    ActionSanitizer,
    DrawdownDisconnector,
    OrdreValide,
)
from jax_arena import (
    DIM_ACTION,
    TDMPC2Planner,
    bridge_pytorch_to_jax,
    initialiser_world_model,
)
from jepa_pipeline import JEPAPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
journal = logging.getLogger("eva.main")

LONGUEUR_FENETRE: int = 128
PERIODE_SEC: float = 1.0
PRIX_REFERENCE: float = 2000.0
DISTANCE_SL: float = 5.0
EQUITY_REFERENCE: float = 100_000.0


@dataclass
class EtatOrchestrateur:
    """État mutable de la boucle d'orchestration.

    Attributes:
        actif: ``False`` dès réception de SIGINT/SIGTERM (arrêt propre).
        moyenne_cem: Warm-start CEM ``(horizon, dim_action)`` ou ``None``.
        ticks: Nombre de ticks traités.
        ordres_emis: Nombre d'ordres validés émis.
    """

    actif: bool = True
    moyenne_cem: object | None = None
    ticks: int = 0
    ordres_emis: int = 0


def flux_marche_simule(longueur: int) -> torch.Tensor:
    """Produit une fenêtre OHLCV synthétique (placeholder de flux réel).

    À remplacer par le connecteur de flux de ticks réel (MT5/cTrader) en
    production ; l'interface ``(1, longueur, 5)`` est conservée.

    Args:
        longueur: Nombre de barres de la fenêtre.

    Returns:
        Tenseur OHLCV ``(1, longueur, 5)`` float32.
    """
    rendements = torch.randn(1, longueur) * 0.001
    close = PRIX_REFERENCE * torch.cumprod(1.0 + rendements, dim=1)
    ouvert = torch.cat([close[:, :1], close[:, :-1]], dim=1)
    amplitude = close * (torch.rand(1, longueur) * 0.002)
    haut = torch.maximum(ouvert, close) + amplitude
    bas = torch.minimum(ouvert, close) - amplitude
    volume = torch.rand(1, longueur) * 1_000.0 + 100.0
    return torch.stack([ouvert, haut, bas, close, volume], dim=-1).float()


class OrchestrateurEVA:
    """Boucle maître : JEPA -> DLPack -> CEM -> Sanitizer -> Disjoncteur.

    Attributes:
        pipeline: Encodeur JEPA (GPU 0).
        planner: Planificateur CEM (GPU 1).
        sanitizer: Filtre de risque 1 %.
        disjoncteur: Garde-fou drawdown 4 %.
        device_jax: Device JAX cible (dernier GPU disponible).
        etat: État mutable de la boucle.
    """

    def __init__(
        self,
        device_pipeline: str = "cuda:0",
        chemin_journal_disjoncteur: str = "logs/disjoncteur.jsonl",
    ) -> None:
        """Construit la chaîne complète et place chaque bloc sur son GPU.

        Args:
            device_pipeline: Device PyTorch de l'encodeur (GPU 0).
            chemin_journal_disjoncteur: Fichier JSONL du disjoncteur.

        Raises:
            RuntimeError: Si aucun GPU JAX n'est disponible.
        """
        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        if not gpus:
            raise RuntimeError("GPU JAX requis pour l'orchestrateur")
        self.device_jax = gpus[-1]
        self.pipeline = JEPAPipeline(device=device_pipeline)
        cle = jax.random.PRNGKey(0)
        self.planner = TDMPC2Planner(initialiser_world_model(cle))
        self.sanitizer = ActionSanitizer()
        self.disjoncteur = DrawdownDisconnector(
            chemin_journal=chemin_journal_disjoncteur
        )
        self.etat = EtatOrchestrateur()
        journal.info(
            "Orchestrateur prêt : pipeline=%s, JAX=%s", device_pipeline, self.device_jax
        )

    def _emettre_ordre(self, ordre: OrdreValide) -> None:
        """Transmet l'ordre validé (placeholder d'exécution courtier).

        Args:
            ordre: Ordre conforme issu du sanitizer.
        """
        self.etat.ordres_emis += 1
        journal.info(
            "ORDRE #%d : dir=%+d lot=%.2f SL=%.2f TP=%.2f (%s)",
            self.etat.ordres_emis,
            ordre.direction,
            ordre.lot,
            ordre.stop_loss,
            ordre.take_profit,
            ordre.raison,
        )

    def tick(self) -> None:
        """Exécute un cycle complet de la chaîne décisionnelle.

        Le disjoncteur est vérifié en premier : s'il a sauté, aucun calcul
        de signal n'est effectué et le cycle est court-circuité.

        Raises:
            ValueError: Si les formes inter-blocs sont incohérentes.
        """
        if not self.disjoncteur.autoriser_ordre():
            rapport = self.disjoncteur.verifier()
            if rapport.declenche:
                journal.critical("%s", rapport.message)
            return

        ohlcv = flux_marche_simule(LONGUEUR_FENETRE)
        latents = self.pipeline.encoder(ohlcv)
        latent_dernier = latents[0, -1, :].contiguous()
        latent_jax = bridge_pytorch_to_jax(latent_dernier, self.device_jax)

        cle = jax.random.PRNGKey(self.etat.ticks)
        action, moyenne = self.planner.planifier(
            cle, latent_jax, moyenne_init=self.etat.moyenne_cem  # type: ignore[arg-type]
        )
        self.etat.moyenne_cem = moyenne

        prix_actuel = float(ohlcv[0, -1, 3])
        signal = np.asarray(action, dtype=np.float64)
        if signal.size < DIM_ACTION:
            raise ValueError(f"signal {signal.shape} trop court")

        ordre = self.sanitizer.sanitiser(
            signal=signal,
            equity=EQUITY_REFERENCE,
            prix=prix_actuel,
            distance_sl=DISTANCE_SL,
        )
        if ordre.direction != 0 and ordre.lot >= self.sanitizer.limites.lot_min:
            self._emettre_ordre(ordre)
        elif ordre.direction != 0:
            journal.warning(
                "Ordre ignoré : lot %.2f < lot_min (risque insuffisant)", ordre.lot
            )
        self.etat.ticks += 1

    def executer(self) -> None:
        """Boucle infinie jusqu'à SIGINT/SIGTERM (arrêt propre)."""

        def arreter(sig: int, _frame: object) -> None:
            journal.warning("Signal %d reçu — arrêt propre demandé", sig)
            self.etat.actif = False

        signal.signal(signal.SIGINT, arreter)
        signal.signal(signal.SIGTERM, arreter)

        journal.info("Boucle démarrée (Ctrl+C pour arrêter)")
        while self.etat.actif:
            debut = time.perf_counter()
            try:
                self.tick()
            except torch.cuda.OutOfMemoryError:
                journal.critical("OOM GPU — vidage du cache et pause")
                torch.cuda.empty_cache()
                time.sleep(5.0)
            except ValueError as exc:
                journal.error("Incohérence de forme/valeur : %s", exc)
            except RuntimeError as exc:
                journal.error("Erreur runtime pipeline : %s", exc)
            ecoule = time.perf_counter() - debut
            time.sleep(max(0.0, PERIODE_SEC - ecoule))

        journal.info(
            "Arrêt : %d ticks, %d ordres émis", self.etat.ticks, self.etat.ordres_emis
        )


def main() -> None:
    """Point d'entrée : construit et lance l'orchestrateur."""
    OrchestrateurEVA().executer()


if __name__ == "__main__":
    main()
