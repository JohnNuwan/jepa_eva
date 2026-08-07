"""Strategie QualiSwing — 3x M15 confirmation + H1/H4 trend + trailing stop.

Fonctionnement :
1. Requiert 3 signaux M15 JEPA consecutifs dans la meme direction
2. Verifie l'alignement des tendances H1 et H4 (EMA20/EMA50)
3. Trailing stop adaptatif (ATR-based)
4. Max 1% risque par trade
"""

import sys, os, json, time, logging
from pathlib import Path

sys.path.insert(0, "/home/aza/projects/jepa_eva")

from main import OrchestrateurEVA, flux_marche_reel, LONGUEUR_FENETRE, calculer_atr, EQUITY_REFERENCE, MULTIPLICATEUR_ATR_SL
from multi_tf import check_mtf_for_jepa, log_mtf
from action_sanitizer import OrdreValide

journal = logging.getLogger("eva.strategy.quali")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s", stream=sys.stdout)


def fetch_ohlcv(symbol: str, bars: int = 64, tf: str = "H1") -> list[dict]:
    """Recupere des bougies depuis le MT5 Bridge avec timeframe configurable."""
    import urllib.request
    url = f"http://192.168.1.6:8765/ohlcv/{symbol}/{bars}/{tf}"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        return data.get("bars", [])
    except Exception as e:
        journal.warning("Erreur fetch %s %s %s: %s", symbol, bars, tf, e)
        return []


def calculer_ema(closes: list[float], periode: int) -> list[float]:
    """Calcule l'EMA."""
    if len(closes) < periode:
        return []
    k = 2.0 / (periode + 1)
    ema = [sum(closes[:periode]) / periode]
    for i in range(periode, len(closes)):
        ema.append(closes[i] * k + ema[-1] * (1 - k))
    return ema


def analyser_tendance(bars: list[dict]) -> int:
    """Analyse la tendance. Retourne +1 (haussiere), -1 (baissiere), 0 (neutre)."""
    if len(bars) < 30:
        return 0
    closes = [b["close"] for b in bars]
    ema20 = calculer_ema(closes, 20)
    ema50 = calculer_ema(closes, 50)
    if not ema20 or not ema50:
        return 0
    last_ema20 = ema20[-1]
    last_ema50 = ema50[-1]
    if last_ema20 > last_ema50 * 1.002:
        return 1
    elif last_ema20 < last_ema50 * 0.998:
        return -1
    return 0


class OrchestrateurQualiSwing(OrchestrateurEVA):
    """Orchestrateur avec confirmation triple + alignement H1/H4 + trailing stop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_buffer: list[int] = []
        self.buffer_size = 3

    def tick(self) -> None:
        if not self.disjoncteur.autoriser_ordre():
            rapport = self.disjoncteur.verifier()
            if rapport.declenche:
                journal.critical("%s", rapport.message)
            return

        # 1. Pipeline JEPA
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

        # 2. Buffer signaux consecutifs
        if direction != 0:
            if not self.signal_buffer or direction == self.signal_buffer[-1]:
                self.signal_buffer.append(direction)
            else:
                self.signal_buffer = [direction]

        if len(self.signal_buffer) < self.buffer_size:
            journal.info("Signal %s — buffer %d/%d",
                         "BUY" if direction > 0 else "SELL" if direction < 0 else "NEUTRE",
                         len(self.signal_buffer), self.buffer_size)
            self.etat.ticks += 1
            time.sleep(1.0)
            return

        # 3. Multi-timeframe alignment check (H4/H1/M15/M5)
        direction_int = 1 if direction > 0 else -1
        mtf_result = check_mtf_for_jepa(self.symbole, direction_int)
        log_mtf(journal, direction_int, mtf_result)

        if not mtf_result["allowed"]:
            self.signal_buffer = []
            self.etat.ticks += 1
            time.sleep(1.0)
            return

        # 4. Trailing stop adaptatif
        try:
            positions = self.connecteur.positions()
            pos_sym = [p for p in positions if p.get("symbol") == self.symbole]
            for pos in pos_sym:
                ptype = pos.get("type")
                current_sl = pos.get("sl", 0)
                pos_price = pos.get("open_price", 0)
                current_price = pos.get("current_price", pos_price)
                profit = pos.get("profit", 0)
                ticket = pos.get("ticket")

                if ptype == "BUY" and current_price > pos_price and profit > 0:
                    new_sl = round(current_price - atr * 2.0, 5)
                    if new_sl > current_sl:
                        self.connecteur.modifier_position(ticket=ticket, sl=new_sl, tp=0)
                        journal.info("TRAILING BUY: SL %.5f->%.5f (p=%.2f$)", current_sl, new_sl, profit)
                elif ptype == "SELL" and current_price < pos_price and profit > 0:
                    new_sl = round(current_price + atr * 2.0, 5)
                    if new_sl < current_sl or current_sl == 0:
                        self.connecteur.modifier_position(ticket=ticket, sl=new_sl, tp=0)
                        journal.info("TRAILING SELL: SL %.5f->%.5f (p=%.2f$)", current_sl, new_sl, profit)
        except Exception as e:
            journal.warning("Erreur trailing stop: %s", e)

        # 5. Risque max 1%
        balance = 10000.0
        try:
            compte = self.connecteur.compte()
            balance = float(compte.get("balance", 10000.0))
        except Exception:
            pass
        max_risk = balance * 0.01
        lot = ordre_brut.lot
        risk_estime = lot * distance_sl * 1000
        if risk_estime > max_risk:
            lot = max(lot * max_risk / risk_estime, self.sanitizer.limites.lot_min)
            journal.info("Risque %.2f$ > 1%% (%.2f$) — lot reduit a %.2f", risk_estime, max_risk, lot)

        if direction != 0 and lot >= self.sanitizer.limites.lot_min:
            sl = ordre_brut.stop_loss
            ordre = OrdreValide(direction, lot, sl, 0.0, ordre_brut.conforme,
                                f"QualiSwing buf={len(self.signal_buffer)}")
            self._emettre_ordre(ordre)
            self.signal_buffer = []

        self.etat.ticks += 1
        time.sleep(1.0)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--max-pos", type=int, default=2)
    args = parser.parse_args()
    SYMBOLE = args.symbol
    JEPA_PATH = f"checkpoints_jepa/jepa_final_{SYMBOLE}_m15.pt"
    WM_PATH = f"checkpoints_wm/world_model_{SYMBOLE}_m15.npz"
    journal.info("Demarrage QualiSwing: %s (max-pos=%d)", SYMBOLE, args.max_pos)
    OrchestrateurQualiSwing(checkpoint_jepa=JEPA_PATH, world_model=WM_PATH, symbole=SYMBOLE).executer()


if __name__ == "__main__":
    main()
