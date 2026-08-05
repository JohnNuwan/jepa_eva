"""Strategie CounterCorr — Correlation XAUUSD/US30 + GPU 1.

Fonctionnement :
1. Recupere les donnees OHLCV XAUUSD et US30.cash simultanement
2. Calcule le coefficient de correlation de Pearson entre les rendements
3. N'entre que lorsque la correlation < -0.7 (divergence)
4. GPU 1 (CUDA_VISIBLE_DEVICES=1)
"""

import sys, os, json, time, logging, math
from pathlib import Path

sys.path.insert(0, "/home/aza/projects/jepa_eva")

from main import OrchestrateurEVA, flux_marche_reel, LONGUEUR_FENETRE, calculer_atr, EQUITY_REFERENCE, MULTIPLICATEUR_ATR_SL
from action_sanitizer import OrdreValide

journal = logging.getLogger("eva.strategy.countercorr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s", stream=sys.stdout)


def fetch_ohlcv_bars(symbol: str, bars: int = 128, tf: str = "M15") -> list[float]:
    """Recupere les prix de cloture depuis le MT5 Bridge."""
    import urllib.request
    url = f"http://192.168.1.6:8765/ohlcv/{symbol}/{bars}/{tf}"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        closes = [b["close"] for b in data.get("bars", [])]
        return closes
    except Exception as e:
        journal.warning("Erreur fetch %s %s %s: %s", symbol, bars, tf, e)
        return []


def calculer_correlation(prix_a: list[float], prix_b: list[float]) -> float:
    """Calcule le coefficient de correlation de Pearson entre deux series."""
    if len(prix_a) < 30 or len(prix_b) < 30:
        return 0.0
    n = min(len(prix_a), len(prix_b))
    prix_a = prix_a[-n:]
    prix_b = prix_b[-n:]

    # Rendements logarithmiques
    rend_a = [math.log(prix_a[i] / prix_a[i - 1]) for i in range(1, n)]
    rend_b = [math.log(prix_b[i] / prix_b[i - 1]) for i in range(1, n)]

    # Pearson
    n_r = len(rend_a)
    if n_r < 20:
        return 0.0
    sum_a = sum(rend_a)
    sum_b = sum(rend_b)
    sum_ab = sum(a * b for a, b in zip(rend_a, rend_b))
    sum_a2 = sum(a * a for a in rend_a)
    sum_b2 = sum(b * b for b in rend_b)

    numerateur = n_r * sum_ab - sum_a * sum_b
    denominateur = math.sqrt((n_r * sum_a2 - sum_a * sum_a) * (n_r * sum_b2 - sum_b * sum_b))

    if denominateur == 0:
        return 0.0
    return numerateur / denominateur


class OrchestrateurCounterCorr(OrchestrateurEVA):
    """Orchestrateur avec filtre de correlation XAUUSD/US30."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tick(self) -> None:
        if not self.disjoncteur.autoriser_ordre():
            rapport = self.disjoncteur.verifier()
            if rapport.declenche:
                journal.critical("%s", rapport.message)
            return

        # 1. Pipeline JEPA normal
        ohlcv = flux_marche_reel(LONGUEUR_FENETRE, self.symbole)
        latents = self.pipeline.encoder(ohlcv)
        latent_dernier = latents[0, -1, :].contiguous()
        import jax, jax.numpy as jnp, numpy as np
        from jax_arena import DIM_ACTION, bridge_pytorch_to_jax
        latent_jax = bridge_pytorch_to_jax(latent_dernier, self.device_jax)
        cle = jax.random.PRNGKey(self.etat.ticks)
        action, moyenne = self.planner.planifier(cle, latent_jax, moyenne_init=self.etat.moyenne_cem)
        self.etat.moyenne_cem = moyenne
        prix_actuel = float(ohlcv[0, -1, 3])
        signal = np.asarray(action, dtype=np.float64)

        atr = calculer_atr(ohlcv)
        distance_sl = max(atr * MULTIPLICATEUR_ATR_SL, 1.0)
        ordre_brut = self.sanitizer.sanitiser(signal=signal, equity=EQUITY_REFERENCE, prix=prix_actuel, distance_sl=distance_sl)
        direction = ordre_brut.direction

        # 2. Correlation XAUUSD / US30.cash
        closes_us30 = fetch_ohlcv_bars("US30.cash", bars=128, tf="M15")
        closes_xau = fetch_ohlcv_bars(self.symbole, bars=128, tf="M15")

        if not closes_us30 or not closes_xau:
            journal.info("Donnees OHLCV indisponibles pour correlation — skip")
            self.etat.ticks += 1
            time.sleep(1.0)
            return

        corr = calculer_correlation(closes_xau, closes_us30)
        journal.info("Correlation XAUUSD/US30 = %.4f", corr)

        # 3. Ne trader que si correlation < -0.7 (divergence)
        if corr >= -0.7:
            journal.info("Correlation %.4f >= -0.7 — pas de divergence, skip", corr)
            self.etat.ticks += 1
            time.sleep(1.0)
            return

        journal.info("Correlation %.4f < -0.7 — divergence detectee!", corr)

        # 4. Emettre l'ordre normal
        if direction != 0 and ordre_brut.lot >= self.sanitizer.limites.lot_min:
            lot = min(ordre_brut.lot, 0.05)
            sl = ordre_brut.stop_loss
            tp = ordre_brut.take_profit
            ordre = OrdreValide(direction, lot, sl, tp, ordre_brut.conforme,
                                f"CounterCorr corr={corr:.3f}")
            self._emettre_ordre(ordre)

        self.etat.ticks += 1
        time.sleep(1.0)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--max-pos", type=int, default=2)
    args = parser.parse_args()
    SYMBOLE = args.symbol
    JEPA_PATH = f"checkpoints_jepa/jepa_final_{SYMBOLE}_m15.pt"
    WM_PATH = f"checkpoints_wm/world_model_{SYMBOLE}_m15.npz"
    journal.info("Demarrage CounterCorr: %s (max-pos=%d)", SYMBOLE, args.max_pos)
    OrchestrateurCounterCorr(checkpoint_jepa=JEPA_PATH, world_model=WM_PATH, symbole=SYMBOLE).executer()


if __name__ == "__main__":
    main()
