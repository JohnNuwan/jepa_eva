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

from datetime import datetime
import json

import logging
import signal
import sys
import time
import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from action_sanitizer import (
    ActionSanitizer,
    DrawdownDisconnector,
    OrdreValide,
)
from jax_arena import (
    DIM_ACTION,
    ParametresWorldModel,
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
MULTIPLICATEUR_ATR_SL: float = 2.0  # SL = 2 × ATR (volatilité réelle)


def calculer_atr(ohlcv: torch.Tensor, periode: int = 14) -> float:
    """Calcule l'Average True Range sur la fenêtre courante.

    L'ATR mesure la volatilité réelle de l'actif : le stop loss s'en inspire
    (SL = multiplicateur × ATR) pour éviter d'être stoppé par le bruit.

    Args:
        ohlcv: Tenseur ``(1, T, 5)`` avec canaux [O, H, L, C, V].
        periode: Période de lissage ATR.

    Returns:
        ATR en points de prix (float).
    """
    haut = ohlcv[0, :, 1]
    bas = ohlcv[0, :, 2]
    cloture_prec = torch.cat([ohlcv[0, :1, 3], ohlcv[0, :-1, 3]])
    tr = torch.maximum(
        haut - bas,
        torch.maximum((haut - cloture_prec).abs(), (bas - cloture_prec).abs()),
    )
    return float(tr[-periode:].mean())


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


def flux_marche_reel(longueur: int, symbole: str = "XAUUSD"):
    """Recupere les vraies donnees OHLCV depuis le MT5 Bridge."""
    import urllib.request
    try:
        url = "http://192.168.1.6:8765/ohlcv/" + symbole + "/" + str(longueur) + "/M15"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        bars = data.get("bars", [])
        if len(bars) >= 2:
            ouv = torch.tensor([b["open"] for b in bars], dtype=torch.float32).unsqueeze(0)
            haut = torch.tensor([b["high"] for b in bars], dtype=torch.float32).unsqueeze(0)
            bas = torch.tensor([b["low"] for b in bars], dtype=torch.float32).unsqueeze(0)
            clo = torch.tensor([b["close"] for b in bars], dtype=torch.float32).unsqueeze(0)
            vol = torch.ones_like(clo) * 0.01
            ref = clo[0, 0].clamp(min=0.0001)
            return torch.stack([ouv/ref, haut/ref, bas/ref, clo/ref, vol], dim=2).float()
    except Exception as e:
        logging.getLogger("eva.main").warning("OHLCV reel indisponible - fallback: %s", e)
    rendements = torch.randn(1, longueur) * 0.001
    close = 50000.0 * torch.cumprod(1.0 + rendements, dim=1)
    ouvert = torch.cat([close[:, :1], close[:, :-1]], dim=1)
    amplitude = close * (torch.rand(1, longueur) * 0.002)
    haut = torch.maximum(ouvert, close) + amplitude
    bas = torch.minimum(ouvert, close) - amplitude
    return torch.stack([ouvert, haut, bas, close, torch.ones_like(close) * 0.01], dim=2).float()

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
        checkpoint_jepa: str = "checkpoints_jepa/jepa_final_XAUUSD_m15.pt",
        world_model: str = "checkpoints_wm/world_model_XAUUSD_m15.npz",
        symbole: str = "US30.cash",
    ) -> None:
        """Construit la chaîne complète et place chaque bloc sur son GPU.

        Charge l'encodeur JEPA pré-entraîné et le world model entraîné si
        disponibles (sinon poids aléatoires avec avertissement).

        Args:
            device_pipeline: Device PyTorch de l'encodeur (GPU 0).
            chemin_journal_disjoncteur: Fichier JSONL du disjoncteur.
            checkpoint_jepa: Checkpoint de l'encodeur JEPA entraîné.
            world_model: Poids du world model GRU entraîné.

        Raises:
            RuntimeError: Si aucun GPU JAX n'est disponible.
        """
        gpus = [d for d in jax.devices() if d.platform == "gpu"]
        if not gpus:
            raise RuntimeError("GPU JAX requis pour l'orchestrateur")
        self.device_jax = gpus[-1]
        self.pipeline = JEPAPipeline(device=device_pipeline)

        # Encodeur JEPA pré-entraîné.
        if Path(checkpoint_jepa).is_file():
            ckpt = torch.load(checkpoint_jepa, map_location=device_pipeline, weights_only=False)
            self.pipeline.modele.encodeur_online.load_state_dict(ckpt["encodeur"])
            self.pipeline.normalisateur.load_state_dict(ckpt["normalisateur"])
            self.pipeline.modele.eval()
            journal.info("Encodeur JEPA chargé (perte=%.5f)", ckpt.get("perte_finale", float("nan")))
        else:
            journal.warning("Checkpoint JEPA absent (%s) — encodeur ALÉATOIRE", checkpoint_jepa)

        # World model entraîné pour le CEM.
        if Path(world_model).is_file():
            donnees = np.load(world_model)
            feuilles = [jnp.asarray(donnees[f"p{i}"]) for i in range(6)]
            params_wm = ParametresWorldModel(*feuilles)
            journal.info("World model entraîné chargé : %s", world_model)
        else:
            params_wm = initialiser_world_model(jax.random.PRNGKey(0))
            journal.warning("World model absent (%s) — CEM ALÉATOIRE", world_model)
        self.planner = TDMPC2Planner(params_wm)

        self.sanitizer = ActionSanitizer()
        self.symbole = symbole
        from connecteur_mt5 import creer_connecteur
        self.connecteur = creer_connecteur('auto')
        self.disjoncteur = DrawdownDisconnector(
            chemin_journal=chemin_journal_disjoncteur
        )
        self.etat = EtatOrchestrateur()
        journal.info(
            "Orchestrateur prêt : pipeline=%s, JAX=%s", device_pipeline, self.device_jax
        )

    def _emettre_ordre(self, ordre: OrdreValide) -> None:
        """Transmet l'ordre validé au MT5 Bridge."""
        self.etat.ordres_emis += 1
        
        # Gestion intelligente des positions multiples
        MAX_POS = 5  # max positions par symbole
        DAILY_TARGET = 200.0  # objectif journalier en euros
        
        try:
            positions = self.connecteur.positions()
            pos_sym = [p for p in positions if p.get("symbol") == self.symbole]
            buys = [p for p in pos_sym if p.get("type") == "BUY"]
            sells = [p for p in pos_sym if p.get("type") == "SELL"]
            existing = buys if ordre.direction > 0 else sells
            
            # Verifier limite de positions
            if len(existing) >= MAX_POS:
                journal.info("Max %d %s %s atteint - ordre ignore", MAX_POS, 
                    "BUY" if ordre.direction > 0 else "SELL", self.symbole)
                return
            
            # Position #2+ seulement si la 1ere est en profit
            if len(existing) >= 1:
                profit_total = sum(p.get("profit", 0) for p in existing)
                if profit_total < 1.0:
                    journal.info("Position #%d %s pas assez profitable (%.2f$) - attente",
                        len(existing)+1, self.symbole, profit_total)
                    return
                journal.info("Position #%d %s autorisee (profit=%.2f$)", 
                    len(existing)+1, self.symbole, profit_total)
            
            # Anti-hedge: verifie chaque position opposee individuellement
            opposite = sells if ordre.direction > 0 else buys
            for p_opp in opposite:
                if p_opp.get("profit", 0) < 5.0:
                    journal.info("Hedge bloque: position opposee %s non securisee (profit=%.2f$)",
                        self.symbole, p_opp.get("profit", 0))
                    return
            
        # Verifier le profit journalier total
            try:
                import urllib.request
                req = urllib.request.Request("http://192.168.1.6:8765/history")
                with urllib.request.urlopen(req, timeout=3) as r:
                    hist = json.loads(r.read().decode())
                today = datetime.now().strftime("%Y-%m-%d")
                daily_profit = sum(d.get("profit", 0) for d in hist.get("deals", []) if today in d.get("close_time", ""))
                
                if daily_profit >= DAILY_TARGET:
                    journal.info("Objectif %.0f$ atteint (%.2f$) - lots reduits a 0.02", 
                        DAILY_TARGET, daily_profit)
                    ordre = OrdreValide(ordre.direction, 0.02, 
                        ordre.stop_loss, ordre.take_profit, ordre.conforme, ordre.raison)
                elif daily_profit > 100 and ordre.lot > 0.02:
                    journal.info("Profit journalier %.2f$ > 100$ - lot reduit de moitie", daily_profit)
                    ordre = OrdreValide(ordre.direction, max(ordre.lot / 2, 0.02), 
                        ordre.stop_loss, ordre.take_profit, ordre.conforme, ordre.raison)
            except Exception as e:
                journal.warning("Erreur verification daily P/L: %s", e)
                
        except Exception as e:
            journal.warning("Erreur verification positions: %s", e)
        
        # Ajuster SL/TP au prix actuel du marche
        try:
            tick = self.connecteur.tick(self.symbole)
            if tick and tick.bid > 0:
                est_achat = ordre.direction > 0
                prix = tick.ask if est_achat else tick.bid
                # Lots adaptes par symbole (risque ~0.3-0.5% par trade)
                LOT_MAX_PAR_SYMBOLE = {
                    "EURUSD": 0.10, "GBPUSD": 0.10, "USDJPY": 0.10,
                    "US30.cash": 0.10, "US500.cash": 0.10, "US100.cash": 0.10,
                    "XAUUSD": 0.05, "GER40.cash": 0.10
                }
                lot = min(ordre.lot, LOT_MAX_PAR_SYMBOLE.get(self.symbole, 0.05))
                SL_DIST = {
                    "EURUSD": 0.0050, "GBPUSD": 0.0050, "USDJPY": 0.80, "USDJPY.cash": 0.80,
                    "XAUUSD": 12.0, "XAGUSD": 1.50,
                    "US30.cash": 500.0, "US100.cash": 600.0, "US500.cash": 100.0,
                    "GER40.cash": 300.0, "FRA40.cash": 200.0,
                }
                dist_sl = SL_DIST.get(self.symbole, prix * 0.01)
                dist_tp = dist_sl * 4  # 4x SL pour laisser le split se faire
                sl = round(prix - dist_sl, 2) if est_achat else round(prix + dist_sl, 2)
                tp = round(prix + dist_tp, 2) if est_achat else round(prix - dist_tp, 2)
                ordre = OrdreValide(ordre.direction, lot, sl, tp, ordre.conforme, ordre.raison)
        except Exception as e:
            journal.warning("Erreur ajustement SL/TP: %s", e)
        
        # Envoyer l'ordre
        try:
            resultat = self.connecteur.envoyer_ordre(
                symbole=self.symbole,
                direction=ordre.direction,
                lot=ordre.lot,
                sl=ordre.stop_loss,
                tp=ordre.take_profit,
            )
            if resultat.succes:
                journal.info("ORDRE #%d : dir=%+d lot=%.2f SL=%.2f TP=%.2f (%s) ✅ EXECUTE",
                    self.etat.ordres_emis, ordre.direction, ordre.lot,
                    ordre.stop_loss, ordre.take_profit, ordre.raison)
            else:
                journal.warning("ORDRE #%d : dir=%+d lot=%.2f SL=%.2f TP=%.2f (%s) ❌ %s",
                    self.etat.ordres_emis, ordre.direction, ordre.lot,
                    ordre.stop_loss, ordre.take_profit, ordre.raison, resultat.message)
        except Exception as e:
            journal.error("Erreur envoi ordre: %s", e)
        
        # Attendre 60s entre les ordres
        time.sleep(60)

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

        ohlcv = flux_marche_reel(LONGUEUR_FENETRE, self.symbole)
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

        # SL basé sur l'ATR (volatilité réelle) plutôt que fixe : évite les
        # stops prématurés par le bruit sur XAUUSD M15 (ATR ~5-15$).
        atr = calculer_atr(ohlcv)
        distance_sl = max(atr * MULTIPLICATEUR_ATR_SL, 1.0)
        ordre = self.sanitizer.sanitiser(
            signal=signal,
            equity=EQUITY_REFERENCE,
            prix=prix_actuel,
            distance_sl=distance_sl,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="US30.cash")
    parser.add_argument("--max-pos", type=int, default=1)
    args = parser.parse_args()
    SYMBOLE = args.symbol
    JEPA_PATH = f"checkpoints_jepa/jepa_final_{SYMBOLE}_m15.pt"
    WM_PATH = f"checkpoints_wm/world_model_{SYMBOLE}_m15.npz"
    """Point d'entrée : construit et lance l'orchestrateur."""
    OrchestrateurEVA(checkpoint_jepa=JEPA_PATH, world_model=WM_PATH, symbole=SYMBOLE).executer()


if __name__ == "__main__":
    main()
