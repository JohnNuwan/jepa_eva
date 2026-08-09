"""Strategie USDJPY H1 — H1 trend filter (EMA40 via M30 proxy) + trailing stop at 2x SL.

Fonctionnement :
1. Utilise le modele JEPA M15 (checkpoints USDJPY_m15 existants)
2. Filtre H1 (proxy M30) : ne trader que dans la direction de la tendance (EMA40 sur M30 ~ EMA40 H1)
3. Pas de TP fixe — trailing stop actif :
   - Quand profit >= 2x SL_distance, SL → entry + 1 spread (breakeven+)
   - Puis trail SL a 1x ATR derriere le prix
4. Max 2 positions par direction (handled locally + --max-pos 2)
5. Comment trades : "EVA-USDJPY" (via TRADE_COMMENT env var)
"""

import sys, os, json, time, logging
from pathlib import Path

sys.path.insert(0, "/home/aza/projects/jepa_eva")

from main import OrchestrateurEVA, flux_marche_reel, LONGUEUR_FENETRE, calculer_atr, EQUITY_REFERENCE, MULTIPLICATEUR_ATR_SL
from multi_tf import check_mtf_for_jepa, log_mtf
from action_sanitizer import OrdreValide

journal = logging.getLogger("eva.strategy.usdjpy_h1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s", stream=sys.stdout)

# USDJPY specifics
SL_DIST_USDJPY = 0.80       # 80 pips SL distance
SPREAD_USDJPY = 0.03        # ~3 pips spread for USDJPY
ATR_TRAIL_MULT = 1.0        # trail at 1x ATR behind price


def fetch_ohlcv_m30(symbol: str, bars: int = 100) -> list[dict]:
    """Recupere des bougies M30 depuis le MT5 Bridge (proxy H1)."""
    import urllib.request
    url = f"http://192.168.1.6:8765/ohlcv/{symbol}/{bars}/M30"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read().decode())
        return data.get("bars", [])
    except Exception as e:
        journal.warning("Erreur fetch M30 %s: %s", symbol, e)
        return []


def calculer_ema(closes: list[float], periode: int) -> list[float]:
    """Calcule l'EMA (Exponential Moving Average)."""
    if len(closes) < periode:
        return []
    k = 2.0 / (periode + 1)
    ema = [sum(closes[:periode]) / periode]
    for i in range(periode, len(closes)):
        ema.append(closes[i] * k + ema[-1] * (1 - k))
    return ema


def analyser_tendance_h1(bars: list[dict]) -> tuple[int, float]:
    """Analyse la tendance via EMA40 sur M30 (equivalent EMA40 H1).

    Returns:
        (direction, ema): direction = +1 (haussiere), -1 (baissiere), 0 (neutre)
    """
    if len(bars) < 50:
        return 0, 0.0
    closes = [b["close"] for b in bars]
    ema = calculer_ema(closes, 40)  # EMA40 on M30 ~ EMA40 on H1
    if not ema:
        return 0, 0.0
    last_ema = ema[-1]
    last_close = closes[-1]
    # H1 trend: price above EMA = bullish, below = bearish
    threshold = last_ema * 0.001  # 0.1% buffer to avoid whipsaw
    if last_close > last_ema + threshold:
        return 1, last_ema
    elif last_close < last_ema - threshold:
        return -1, last_ema
    return 0, last_ema


class OrchestrateurUSDJPY_H1(OrchestrateurEVA):
    """Orchestrateur USDJPY avec filtre H1 (EMA40) + trailing stop 3.0x SL."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _trailing_stop(self, prix_actuel: float, atr: float) -> None:
        """Trailing stop adaptatif.

        Quand profit >= 2x SL_distance (1.60$), SL place d'abord a
        entry + 1 spread (breakeven lock), puis trail a 1x ATR derriere le prix.
        """
        entry_buffer = SL_DIST_USDJPY * 2  # 1.60 — seuil d'activation du trailing
        trail_dist = max(atr * ATR_TRAIL_MULT, 0.20)  # min 20 pips de trail

        try:
            positions = self.connecteur.positions()
            pos_sym = [p for p in positions if p.get("symbol") == self.symbole]

            for pos in pos_sym:
                ptype = pos.get("type")
                current_sl = pos.get("sl", 0) or 0
                pos_price = pos.get("open_price", 0)
                current_price = pos.get("current_price", pos_price)
                profit = pos.get("profit", 0)
                ticket = pos.get("ticket")

                if ptype == "BUY":
                    if profit >= entry_buffer:
                        breakeven_sl = round(pos_price + SPREAD_USDJPY, 3)
                        if current_sl < breakeven_sl:
                            self.connecteur.modifier_position(ticket=ticket, sl=breakeven_sl, tp=0)
                            journal.info("TRAILING BUY: breakeven SL %.3f->%.3f (p=%.2f$)",
                                         current_sl, breakeven_sl, profit)
                            current_sl = breakeven_sl
                        new_sl = round(current_price - trail_dist, 3)
                        if new_sl > current_sl:
                            self.connecteur.modifier_position(ticket=ticket, sl=new_sl, tp=0)
                            journal.info("TRAILING BUY: SL %.3f->%.3f (p=%.2f$, atr=%.3f)",
                                         current_sl, new_sl, profit, atr)

                elif ptype == "SELL":
                    if profit >= entry_buffer:
                        breakeven_sl = round(pos_price - SPREAD_USDJPY, 3)
                        if current_sl == 0 or current_sl > breakeven_sl:
                            self.connecteur.modifier_position(ticket=ticket, sl=breakeven_sl, tp=0)
                            journal.info("TRAILING SELL: breakeven SL %.3f->%.3f (p=%.2f$)",
                                         current_sl, breakeven_sl, profit)
                            current_sl = breakeven_sl
                        new_sl = round(current_price + trail_dist, 3)
                        if current_sl == 0 or new_sl < current_sl:
                            self.connecteur.modifier_position(ticket=ticket, sl=new_sl, tp=0)
                            journal.info("TRAILING SELL: SL %.3f->%.3f (p=%.2f$, atr=%.3f)",
                                         current_sl, new_sl, profit, atr)

        except Exception as e:
            journal.warning("Erreur trailing stop: %s", e)

    def tick(self) -> None:
        if not self.disjoncteur.autoriser_ordre():
            rapport = self.disjoncteur.verifier()
            if rapport.declenche:
                journal.critical("%s", rapport.message)
            return

        # 1. Pipeline JEPA (M15 — meme checkpoint USDJPY_m15)
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
        ordre_brut = self.sanitizer.sanitiser(
            signal=signal, equity=EQUITY_REFERENCE, prix=prix_actuel, distance_sl=distance_sl
        )
        direction = ordre_brut.direction

        # 2. Trailing stop sur positions existantes (AVANT nouvelle entree)
        self._trailing_stop(prix_actuel, atr)

        # 3. Multi-timeframe alignment check (H4/H1/M15/M5)
        direction_int = 1 if direction > 0 else -1 if direction < 0 else 0
        if direction_int == 0:
            journal.info("Signal M15 neutre — skip")
            self.etat.ticks += 1
            time.sleep(1.0)
            return

        mtf_result = check_mtf_for_jepa(self.symbole, direction_int)
        log_mtf(journal, direction_int, mtf_result)

        if not mtf_result["allowed"]:
            self.etat.ticks += 1
            time.sleep(1.0)
            return
        
        # Extract H4 trend for logging
        trend_h4 = mtf_result["features"]["h4"]["trend_dir"]

        # 4. Limite max 2 positions par direction (local, en plus de --max-pos)
        try:
            positions = self.connecteur.positions()
            pos_sym = [p for p in positions if p.get("symbol") == self.symbole]
            buys = [p for p in pos_sym if p.get("type") == "BUY"]
            sells = [p for p in pos_sym if p.get("type") == "SELL"]
            existing = buys if direction_int > 0 else sells

            if len(existing) >= 2:
                journal.info("Max 2 %s positions atteint — skip",
                             "BUY" if direction_int > 0 else "SELL")
                self.etat.ticks += 1
                time.sleep(1.0)
                return
        except Exception as e:
            journal.warning("Erreur verification positions: %s", e)

        # 5. Emettre l'ordre avec SL uniquement (pas de TP fixe — trailing stop gere)
        if direction != 0 and ordre_brut.lot >= self.sanitizer.limites.lot_min:
            try:
                tick = self.connecteur.tick(self.symbole)
                if tick and tick.bid > 0:
                    base_price = tick.ask if direction_int > 0 else tick.bid
                else:
                    base_price = prix_actuel
            except Exception:
                base_price = prix_actuel

            if direction_int > 0:
                sl = round(base_price - SL_DIST_USDJPY, 3)
            else:
                sl = round(base_price + SL_DIST_USDJPY, 3)

            # Pas de TP fixe — trailing stop s'en charge dynamiquement
            ordre = OrdreValide(direction, ordre_brut.lot, sl, 0.0, ordre_brut.conforme,
                                f"USDJPY-MTF H4={trend_h4}")
            self._emettre_ordre(ordre)

        self.etat.ticks += 1
        time.sleep(1.0)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="USDJPY")
    parser.add_argument("--max-pos", type=int, default=2)
    args = parser.parse_args()
    SYMBOLE = args.symbol
    JEPA_PATH = f"checkpoints_jepa/jepa_final_{SYMBOLE}_m15.pt"
    WM_PATH = f"checkpoints_wm/world_model_{SYMBOLE}_m15.npz"
    journal.info("Demarrage USDJPY-H1: %s (max-pos=%d)", SYMBOLE, args.max_pos)
    OrchestrateurUSDJPY_H1(
        checkpoint_jepa=JEPA_PATH,
        world_model=WM_PATH,
        symbole=SYMBOLE,
    ).executer()


if __name__ == "__main__":
    main()