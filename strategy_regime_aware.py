"""Stratégie RegimeAware — ADX filter + 5x M15 confirmation + trailing stop.

Fonctionnement :
1. Calcule l'ADX sur H1 (128 bougies H1 depuis le MT5 Bridge)
2. Vérifie ADX > 25 avant toute entrée (régime de tendance forte)
3. Accumule les signaux M15 consécutifs dans le même sens
4. N'entre qu'après 5 signaux M15 consécutifs
5. Trailing stop au lieu de TP fixe
6. Taille de position dynamique selon ADX
"""

import sys, os, json, time, logging
from pathlib import Path

sys.path.insert(0, "/home/aza/projects/jepa_eva")

from main import OrchestrateurEVA, flux_marche_reel, LONGUEUR_FENETRE, calculer_atr, EQUITY_REFERENCE, MULTIPLICATEUR_ATR_SL
from action_sanitizer import OrdreValide

journal = logging.getLogger("eva.strategy.regime")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s", stream=sys.stdout)


def fetch_ohlcv_h1(symbol: str, bars: int = 128) -> list[dict]:
    """Recupere des bougies M30 depuis le MT5 Bridge (proxy H1, TF max dispo)."""
    import urllib.request
    url = f"http://192.168.1.6:8765/ohlcv/{symbol}/{bars}/M30"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        return data.get("bars", [])
    except Exception as e:
        journal.warning("Erreur fetch H1 %s: %s", symbol, e)
        return []


def calculer_adx(bars: list[dict], periode: int = 14) -> float:
    """Calcule l'ADX (Average Directional Index) a partir de bougies OHLC."""
    if len(bars) < periode + 1:
        return 0.0

    hauts = [b["high"] for b in bars]
    bas = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]

    plus_dm = []
    minus_dm = []
    tr_values = []

    for i in range(1, len(bars)):
        up_move = hauts[i] - hauts[i - 1]
        down_move = bas[i - 1] - bas[i]

        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)

        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)

        tr = max(hauts[i] - bas[i], abs(hauts[i] - closes[i - 1]), abs(bas[i] - closes[i - 1]))
        tr_values.append(tr)

    if len(tr_values) < periode:
        return 0.0

    # Wilder smoothing
    atr = sum(tr_values[:periode]) / periode
    sum_plus = sum(plus_dm[:periode]) / periode
    sum_minus = sum(minus_dm[:periode]) / periode

    for i in range(periode, len(tr_values)):
        atr = (atr * (periode - 1) + tr_values[i]) / periode
        sum_plus = (sum_plus * (periode - 1) + plus_dm[i]) / periode
        sum_minus = (sum_minus * (periode - 1) + minus_dm[i]) / periode

    plus_di = 100.0 * sum_plus / atr if atr > 0 else 0.0
    minus_di = 100.0 * sum_minus / atr if atr > 0 else 0.0

    dx = 100.0 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0.0
    return dx


class OrchestrateurRegimeAware(OrchestrateurEVA):
    """Orchestrateur avec filtre ADX + confirmation 5x M15 + trailing stop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_buffer: list[int] = []
        self.buffer_size = 5

    def _fetch_h1_adx(self) -> float:
        bars = fetch_ohlcv_h1(self.symbole, bars=128)
        if not bars:
            return 0.0
        return calculer_adx(bars, periode=14)

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

        # 2. Trailing stop si position active
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
                    new_sl = round(current_price - distance_sl * 1.5, 2)
                    if new_sl > current_sl:
                        self.connecteur.modifier_position(ticket=ticket, sl=new_sl, tp=0)
                        journal.info("TRAILING BUY: SL %.2f->%.2f (p=%.2f$)", current_sl, new_sl, profit)
                elif ptype == "SELL" and current_price < pos_price and profit > 0:
                    new_sl = round(current_price + distance_sl * 1.5, 2)
                    if new_sl < current_sl or current_sl == 0:
                        self.connecteur.modifier_position(ticket=ticket, sl=new_sl, tp=0)
                        journal.info("TRAILING SELL: SL %.2f->%.2f (p=%.2f$)", current_sl, new_sl, profit)
        except Exception as e:
            journal.warning("Erreur trailing stop: %s", e)

        # 3. ADX filter (H1)
        adx = self._fetch_h1_adx()
        if adx < 25:
            journal.info("ADX=%.1f < 25 — skip (pas de tendance)", adx)
            self.signal_buffer = []
            self.etat.ticks += 1
            time.sleep(1.0)
            return

        journal.info("ADX=%.1f >= 25 — tendance forte", adx)

        # 4. Buffer signaux consecutifs
        if direction != 0:
            if not self.signal_buffer or direction == self.signal_buffer[-1]:
                self.signal_buffer.append(direction)
            else:
                self.signal_buffer = [direction]

        if len(self.signal_buffer) < self.buffer_size:
            journal.info("Signal %s — buffer %d/%d, ADX=%.1f",
                         "BUY" if direction > 0 else "SELL" if direction < 0 else "NEUTRE",
                         len(self.signal_buffer), self.buffer_size, adx)
            self.etat.ticks += 1
            time.sleep(1.0)
            return

        # 5. Taille dynamique selon ADX
        lot_base = ordre_brut.lot
        if adx >= 35:
            lot = min(lot_base * 1.5, 0.15)
        else:
            lot = min(lot_base, 0.10)

        if direction != 0 and lot >= self.sanitizer.limites.lot_min:
            sl = ordre_brut.stop_loss
            ordre = OrdreValide(direction, lot, sl, 0.0, ordre_brut.conforme,
                                f"RegimeAware ADX={adx:.1f}")
            self._emettre_ordre(ordre)
            self.signal_buffer = []
        elif direction != 0:
            journal.warning("Ordre ignore: lot %.2f < lot_min", lot)

        self.etat.ticks += 1
        time.sleep(1.0)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="US30.cash")
    parser.add_argument("--max-pos", type=int, default=3)
    args = parser.parse_args()
    SYMBOLE = args.symbol
    JEPA_PATH = f"checkpoints_jepa/jepa_final_{SYMBOLE}_m15.pt"
    WM_PATH = f"checkpoints_wm/world_model_{SYMBOLE}_m15.npz"
    journal.info("Demarrage RegimeAware: %s (max-pos=%d)", SYMBOLE, args.max_pos)
    OrchestrateurRegimeAware(checkpoint_jepa=JEPA_PATH, world_model=WM_PATH, symbole=SYMBOLE).executer()


if __name__ == "__main__":
    main()
