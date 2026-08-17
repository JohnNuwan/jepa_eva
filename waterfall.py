#!/usr/bin/env python3
"""waterfall.py — Chaîne d'interception pour les décisions de trading EVA.
Pattern inspiré de DeepSeek Harness (agent/pre-step waterfall).

Chaque ordre passe par une chaîne de guards. Chaque guard peut :
  - "allow"    : laisser passer (éventuellement en modifiant le contexte)
  - "deny"     : bloquer l'ordre (autoritatif)
  - "rewrite"  : modifier l'ordre puis continuer

La waterfall est testable indépendamment du broker."""
from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

journal = logging.getLogger("eva.waterfall")


@dataclass
class CtxWaterfall:
    """Contexte partagé traversant la chaîne."""
    symbole: str
    direction: int            # +1 BUY / -1 SELL
    lot: float
    stop_loss: float
    take_profit: float
    raison: str = ""
    positions: list = field(default_factory=list)   # positions du symbole
    compte: dict = field(default_factory=dict)
    tick: Optional[Any] = None
    etat_global: dict = field(default_factory=dict)  # état self-awareness / risk
    modifs: list = field(default_factory=list)       # journal des guards qui ont agi
    DENY: bool = False
    RAISON_DENY: str = ""


@dataclass
class ResultatWaterfall:
    """Verdict final de la chaîne."""
    autorise: bool
    ctx: CtxWaterfall
    garde_bloquante: str = ""
    raison: str = ""

    def as_json(self) -> str:
        return json.dumps({
            "autorise": self.autorise,
            "symbole": self.ctx.symbole,
            "direction": self.ctx.direction,
            "lot": self.ctx.lot,
            "garde": self.garde_bloquante,
            "raison": self.raison,
            "modifs": self.ctx.modifs,
        }, ensure_ascii=False)


class Waterfall:
    """Chaîne d'interception style DeepSeek Harness."""

    def __init__(self, nom: str = "trade"):
        self.nom = nom
        self._guards: list[Callable[[CtxWaterfall], str]] = []

    def add(self, guard: Callable[[CtxWaterfall], str]) -> "Waterfall":
        """Ajoute un étage. Le guard retourne 'allow' | 'deny' | 'rewrite'."""
        self._guards.append(guard)
        return self

    def run(self, ctx: CtxWaterfall) -> ResultatWaterfall:
        """Exécute la chaîne. Le premier 'deny' bloque (autoritatif)."""
        for guard in self._guards:
            nom = getattr(guard, "__name__", "guard")
            try:
                verdict = guard(ctx)
            except Exception as e:
                journal.warning("[%s] guard %s a levé une exception: %s — DÉFAUT deny (fail-safe)",
                                self.nom, nom, e)
                return ResultatWaterfall(False, ctx, nom, f"exception guard: {e}")
            if verdict == "deny":
                ctx.DENY = True
                return ResultatWaterfall(False, ctx, nom, ctx.RAISON_DENY or f"bloqué par {nom}")
            if verdict == "rewrite":
                ctx.modifs.append(nom)
                continue
            # allow
            continue
        return ResultatWaterfall(True, ctx)


# ============================================================
# GUARDS DE BASE (anciennes vérifications de _emettre_ordre)
# ============================================================

def gard_max_positions(ctx: CtxWaterfall, max_pos: int = 5) -> str:
    """Limite le nombre de positions par symbole."""
    buys = [p for p in ctx.positions if p.get("type") == "BUY"]
    sells = [p for p in ctx.positions if p.get("type") == "SELL"]
    existing = buys if ctx.direction > 0 else sells
    if len(existing) >= max_pos:
        ctx.RAISON_DENY = f"max {max_pos} {('BUY' if ctx.direction > 0 else 'SELL')} {ctx.symbole} atteint"
        return "deny"
    return "allow"


def gard_profit_avance(ctx: CtxWaterfall, seuil_profit: float = 1.0, commission_par_lot: float = 3.0) -> str:
    """Position #2+ uniquement si la 1ère est en profit."""
    buys = [p for p in ctx.positions if p.get("type") == "BUY"]
    sells = [p for p in ctx.positions if p.get("type") == "SELL"]
    existing = buys if ctx.direction > 0 else sells
    if len(existing) >= 1:
        commission = commission_par_lot * ctx.lot
        profit_total = sum(p.get("profit", 0) for p in existing) - commission * len(existing)
        if profit_total < seuil_profit:
            ctx.RAISON_DENY = f"position #{len(existing)+1} {ctx.symbole} pas assez profitable ({profit_total:.2f}$)"
            return "deny"
    return "allow"


def gard_anti_hedge(ctx: CtxWaterfall, seuil_secu: float = 5.0) -> str:
    """Bloque le hedge si une position opposée est en perte."""
    buys = [p for p in ctx.positions if p.get("type") == "BUY"]
    sells = [p for p in ctx.positions if p.get("type") == "SELL"]
    opposite = sells if ctx.direction > 0 else buys
    for p_opp in opposite:
        if p_opp.get("profit", 0) < seuil_secu:
            ctx.RAISON_DENY = f"hedge bloqué: position opposée non sécurisée (profit={p_opp.get('profit',0):.2f}$)"
            return "deny"
    return "allow"


def gard_daily_target(ctx: CtxWaterfall, bridge_url: str = "http://192.168.1.6:8765",
                      cible: float = 200.0, seuil_reduction: float = 100.0) -> str:
    """Réduit les lots si l'objectif journalier est proche / atteint."""
    try:
        req = urllib.request.Request(f"{bridge_url}/history")
        with urllib.request.urlopen(req, timeout=3) as r:
            hist = json.loads(r.read().decode())
        today = datetime.now().strftime("%Y-%m-%d")
        daily_profit = sum(d.get("profit", 0) for d in hist.get("deals", [])
                           if today in d.get("close_time", ""))
        if daily_profit >= cible:
            ctx.lot = 0.02
            ctx.modifs.append("daily_target:lot_reduit_0.02")
        elif daily_profit > seuil_reduction and ctx.lot > 0.02:
            ctx.lot = max(ctx.lot / 2, 0.02)
            ctx.modifs.append("daily_target:lot_divise_2")
    except Exception as e:
        journal.warning("garde daily_target: %s", e)
    return "allow"


def gard_self_awareness(ctx: CtxWaterfall) -> str:
    """Consulte les décisions de la self-awareness (pauses symbole, budget risque)."""
    st = ctx.etat_global or {}
    # Pause symbole (comme la pause 4h XAUUSD décidée par self-awareness)
    pauses = st.get("pauses", {})
    if ctx.symbole in pauses:
        fin = pauses.get(ctx.symbole, {}).get("fin", "")
        raison = pauses.get(ctx.symbole, {}).get("raison", "pause self-awareness")
        ctx.RAISON_DENY = f"{ctx.symbole} en pause ({raison}) jusqu'à {fin}"
        return "deny"
    # Régime défavorable global → réduire les lots
    regime = st.get("regime", "")
    if regime == "trending_unfavorable":
        ctx.lot = max(ctx.lot * 0.5, 0.01)
        ctx.modifs.append(f"regime:{regime}:lot_*0.5")
    # Budget quotidien épuisé → bloquer nouveaux ordres
    if st.get("budget_epuise"):
        ctx.RAISON_DENY = "budget risque quotidien épuisé"
        return "deny"
    return "allow"


def gard_risk_limits(ctx: CtxWaterfall, perte_jour_max: float = 100.0,
                     drawdown_max: float = 0.10) -> str:
    """Filet de sécurité absolu: perte journalière ou drawdown > seuil → STOP."""
    perte_jour = ctx.etat_global.get("perte_jour", 0.0)
    drawdown = ctx.etat_global.get("drawdown", 0.0)
    if perte_jour <= -perte_jour_max:
        ctx.RAISON_DENY = f"perte journalière {perte_jour:.2f}€ ≤ -{perte_jour_max}€ — STOP"
        return "deny"
    if drawdown >= drawdown_max:
        ctx.RAISON_DENY = f"drawdown {drawdown:.1%} ≥ {drawdown_max:.0%} — STOP"
        return "deny"
    return "allow"


# ============================================================
# FABRICATION
# ============================================================

def creer_waterfall_trading() -> Waterfall:
    """Chaîne standard pour un ordre de trading JEPA."""
    wf = Waterfall("trading")
    wf.add(gard_risk_limits)        # 1. filet de sécurité absolu
    wf.add(gard_self_awareness)     # 2. décisions self-awareness
    wf.add(gard_max_positions)      # 3. position limits
    wf.add(gard_profit_avance)      # 4. progression intelligente
    wf.add(gard_anti_hedge)         # 5. anti-hedge
    wf.add(gard_daily_target)       # 6. gestion objectif journalier
    return wf


# ============================================================
# TEST UNITAIRE INTÉGRÉ
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    tests = []

    def t(nom: str, cond: bool) -> None:
        tests.append(cond)
        print(f"  {'✅' if cond else '❌'} {nom}")

    print("== Tests waterfall ==")

    # 1. Ordre simple autorisé
    wf = creer_waterfall_trading()
    ctx = CtxWaterfall("XAUUSD", 1, 0.05, 4100, 4200, positions=[])
    r = wf.run(ctx)
    t("ordre simple autorisé", r.autorise)

    # 2. Max positions
    wf = creer_waterfall_trading()
    positions = [{"type": "BUY", "profit": 10.0, "symbol": "XAUUSD"} for _ in range(5)]
    ctx = CtxWaterfall("XAUUSD", 1, 0.05, 4100, 4200, positions=positions)
    r = wf.run(ctx)
    t("max positions bloqué", not r.autorise and r.garde_bloquante == "gard_max_positions")

    # 3. Profit avance (positions ok mais pas de profit)
    wf = creer_waterfall_trading()
    positions = [{"type": "BUY", "profit": -2.0, "symbol": "XAUUSD"}]
    ctx = CtxWaterfall("XAUUSD", 1, 0.05, 4100, 4200, positions=positions)
    r = wf.run(ctx)
    t("position #2 sans profit bloqué", not r.autorise and r.garde_bloquante == "gard_profit_avance")

    # 4. Anti-hedge
    wf = creer_waterfall_trading()
    positions = [{"type": "SELL", "profit": 2.0, "symbol": "XAUUSD"}]
    ctx = CtxWaterfall("XAUUSD", 1, 0.05, 4100, 4200, positions=positions)
    r = wf.run(ctx)
    t("hedge non sécurisé bloqué", not r.autorise and r.garde_bloquante == "gard_anti_hedge")

    # 5. Self-awareness pause
    wf = creer_waterfall_trading()
    ctx = CtxWaterfall("XAUUSD", 1, 0.05, 4100, 4200,
                       etat_global={"pauses": {"XAUUSD": {"fin": "22:00", "raison": "294 cycles de perte"}}})
    r = wf.run(ctx)
    t("pause self-awareness bloqué", not r.autorise and r.garde_bloquante == "gard_self_awareness")

    # 6. Risk limits drawdown
    wf = creer_waterfall_trading()
    ctx = CtxWaterfall("XAUUSD", 1, 0.05, 4100, 4200, etat_global={"drawdown": 0.12})
    r = wf.run(ctx)
    t("drawdown >10% bloqué", not r.autorise and r.garde_bloquante == "gard_risk_limits")

    # 7. Régime défavorable → rewrite lot
    wf = creer_waterfall_trading()
    ctx = CtxWaterfall("XAUUSD", 1, 0.05, 4100, 4200, etat_global={"regime": "trending_unfavorable"})
    r = wf.run(ctx)
    t("régime défavorable autorise avec lot réduit", r.autorise and ctx.lot < 0.05)

    print(f"\nRésultat: {sum(tests)}/{len(tests)} tests passés")
    exit(0 if all(tests) else 1)