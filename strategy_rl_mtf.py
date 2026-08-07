#!/usr/bin/env python3
"""EVA-RL v4: Q-learning with Shadow Agent, Self-Learning, and Swap Mechanism

Shadow Agent: parallel simulated agent with its own Q-table.
Self-Learning: analyze past trades to boost/reduce Q-values offline.
Swap: when shadow outperforms live by 10%+, swap Q-tables.
"""
import numpy as np, pandas as pd, json, urllib.request, os, pickle
from datetime import datetime
TF = os.environ.get("TRADE_TF","M5")
if TF == "M1":
    MIN_HOLD = 10
    HOLDING_COST = -0.005
elif TF == "M5":
    MIN_HOLD = 4
    HOLDING_COST = -0.01
else:
    MIN_HOLD = 2
    HOLDING_COST = -0.02

SYMBOL = os.environ.get("TRADE_SYMBOL", "US30.cash")
SYM_SAFE = SYMBOL.replace(chr(46), chr(95))
BRIDGE = "http://192.168.1.6:8765"
COMMENT = "EVA-RL-" + TF
MAX_POS = 3

Q_FILE = f"/home/aza/eva-adam-v2/data/rl_qtable_{SYM_SAFE}_{TF}.pkl"
Q_FILE_SHADOW = f"/home/aza/eva-adam-v2/data/rl_qtable_{SYM_SAFE}_{TF}_shadow.pkl"
LOG = f"/home/aza/eva-adam-v2/logs/rl_mtf_{SYM_SAFE}_{TF}.log"
SWAP_LOG = f"/home/aza/eva-adam-v2/logs/rl_swap_{SYM_SAFE}_{TF}.log"
TRADE_HISTORY = f"/home/aza/eva-adam-v2/data/rl_trades_{SYM_SAFE}.json"
SWAP_STATE_FILE = f"/home/aza/eva-adam-v2/data/rl_swap_state_{SYM_SAFE}_{TF}.json"
SHADOW_STATE_FILE = f"/home/aza/eva-adam-v2/data/rl_shadow_state_{SYM_SAFE}_{TF}.json"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def log(m, f=None):
    t = datetime.now().isoformat()[:19]
    path = f or LOG
    with open(path, "a") as fh:
        fh.write(t + " " + m + chr(10))
    print(t, m)


# ---------------------------------------------------------------------------
# Live Agent
# ---------------------------------------------------------------------------

class RLAgent:
    """Q-learning agent with reward shaping — unchaged interface from v3."""

    def __init__(self, q_file=None, agent_type="live"):
        self.q_table = {}
        self.alpha = 0.05
        self.gamma = 0.99
        self.epsilon = 0.2
        self.recent_rewards = []
        self.trades_count = 0
        self.open_trades = {}       # ticket -> {bars, dir, action, entry_state, entry_price}
        self.last_state = None
        self.q_file = q_file or Q_FILE
        self.agent_type = agent_type
        if os.path.exists(self.q_file):
            with open(self.q_file, "rb") as f:
                self.q_table = pickle.load(f)

    def save(self):
        with open(self.q_file, "wb") as f:
            pickle.dump(self.q_table, f)

    def get_state(self, df, mtf):
        """6-dim state: [H4_trend, H4_atr, H1_rsi, M15_rsi, M15_macd, M5_micro]"""
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values
        if len(c) < 14:
            return (0, 0, 0, 0, 0, 0)
        delta = np.diff(c[-15:])
        gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0.001
        loss = -np.mean(delta[delta < 0]) if np.any(delta < 0) else 0.001
        rsi = 100 - 100 / (1 + gain / loss)
        macd = c[-1] - np.mean(c[-5:])
        atr = np.mean(h[-14:] - l[-14:])
        atr_ratio = atr / np.mean(h[-50:] - l[-50:]) if len(h) > 50 else 0.5
        h4_dir = mtf.get("h4_dir", 0)
        h4_atr = mtf.get("h4_atr", 0)
        h1_rsi = mtf.get("h1_rsi", 50)
        m5_trend = mtf.get("m5_trend", 0)
        return (int(h4_dir), int(h4_atr * 5), int(h1_rsi // 10), int(rsi // 10),
                int(macd / atr * 10) if atr != 0 else 0, int(m5_trend))

    def act(self, state):
        if state not in self.q_table:
            self.q_table[state] = {"buy": 0, "sell": 0, "hold": 0}
        if len(self.recent_rewards) > 20:
            wr = sum(1 for r in self.recent_rewards[-20:] if r > 0) / 20
            self.epsilon = max(0.05, min(0.3, 0.2 * (1 - wr) * 2))
        if np.random.random() < self.epsilon:
            return np.random.choice(["buy", "sell", "hold"])
        return max(self.q_table[state], key=self.q_table[state].get)

    def get_size(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = {"buy": 0, "sell": 0, "hold": 0}
        values = list(self.q_table[state].values())
        confidence = self.q_table[state][action] - np.median(values) if len(values) > 0 else 0
        size = np.clip(abs(confidence) / 1.0, 0.01, 0.1)
        return round(size, 2)

    def update(self, s, a, r, ns):
        for st in [s, ns]:
            if st not in self.q_table:
                self.q_table[st] = {"buy": 0, "sell": 0, "hold": 0}
        self.recent_rewards.append(r)
        self.trades_count += 1
        mf = max(self.q_table[ns].values())
        self.q_table[s][a] += self.alpha * (r + self.gamma * mf - self.q_table[s][a])
        self.save()

    def update_from_open_positions(self, mtf, atr, current_state):
        """Calculate intermediate rewards from open positions every 5min cycle."""
        if not hasattr(self, 'open_trades'):
            self.open_trades = {}

        try:
            req = urllib.request.urlopen(BRIDGE + "/positions", timeout=5)
            pos = json.loads(req.read().decode()).get("positions", [])
            my_pos = [p for p in pos if COMMENT in p.get("comment", "")]
        except Exception as e:
            log(f"ERR interim fetch: {e}")
            return 0

        h4_dir = mtf.get("h4_dir", 0)
        tracked = set()
        total_reward = 0.0
        count = 0

        for p in my_pos:
            ticket = p["ticket"]
            tracked.add(ticket)

            ptype = p.get("type", "sell")
            if isinstance(ptype, int):
                ptype = "buy" if ptype == 0 else "sell"
            direction = ptype.lower()
            dir_sign = 1 if direction == "buy" else -1

            if ticket not in self.open_trades:
                self.open_trades[ticket] = {
                    "bars": 0,
                    "dir": direction,
                    "action": direction,
                    "entry_state": current_state,
                    "entry_price": float(p.get("open_price", 0))
                }
            self.open_trades[ticket]["bars"] += 1
            bars = self.open_trades[ticket]["bars"]

            profit = float(p.get("profit", 0))
            float_reward = profit / atr if abs(atr) > 1e-10 else 0.0

            aligned = (dir_sign == 1 and h4_dir == 1) or (dir_sign == -1 and h4_dir == -1)
            dir_bonus = 1.0 if aligned else -1.0

            holding_cost = HOLDING_COST * bars
            combined = float_reward + dir_bonus + holding_cost

            s_entry = self.open_trades[ticket]["entry_state"]
            a_entry = self.open_trades[ticket]["action"]

            if s_entry not in self.q_table:
                self.q_table[s_entry] = {"buy": 0, "sell": 0, "hold": 0}

            mf = max(self.q_table[current_state].values()) if current_state in self.q_table else 0
            self.q_table[s_entry][a_entry] += self.alpha * (
                combined + self.gamma * mf - self.q_table[s_entry][a_entry]
            )

            total_reward += combined
            count += 1

            log(f"INTERIM tkt={ticket} {direction} "
                f"flP&L={float_reward:+.3f} "
                f"dir={dir_bonus:.0f} "
                f"hold={holding_cost:.2f} "
                f"tot={combined:+.3f} "
                f"bars={bars} "
                f"state={s_entry}")

        for ticket in list(self.open_trades.keys()):
            if ticket not in tracked:
                del self.open_trades[ticket]

        if count > 0:
            self.save()
            log(f"INTERIM total trades={count} sum_reward={total_reward:+.3f}")

        return count


# ---------------------------------------------------------------------------
# Shadow Agent (simulated trading)
# ---------------------------------------------------------------------------

class ShadowAgent(RLAgent):
    """Clone of RLAgent with separate Q-table — trades are SIMULATED, not real.

    Maintains virtual balance, virtual positions, and a log of virtual trades.
    Uses the same state/action space but never sends orders to the bridge.
    """

    def __init__(self):
        super().__init__(q_file=Q_FILE_SHADOW, agent_type="shadow")
        self.virtual_balance = 0.0
        self.virtual_positions = {}       # vid -> {dir, entry_price, volume, entry_state, action, bars, entry_time}
        self.virtual_pnl = 0.0            # cumulative P&L from all closed virtual trades
        self.virtual_trades_log = []      # list of closed trade dicts
        self.next_id = 0

    def simulate_open(self, action, state, current_price, atr):
        """Simulate opening a trade. Returns vid or None."""
        if action == "hold":
            return None
        if len(self.virtual_positions) >= MAX_POS:
            return None
        size = self.get_size(state, action)
        self.next_id += 1
        vid = f"shadow_{self.next_id}"
        self.virtual_positions[vid] = {
            "dir": action,
            "entry_price": current_price,
            "volume": size,
            "entry_state": state,
            "action": action,
            "bars": 0,
            "entry_time": datetime.now().isoformat()[:19]
        }
        # Ensure state is in Q-table and persist immediately
        if state not in self.q_table:
            self.q_table[state] = {"buy": 0, "sell": 0, "hold": 0}
        self.save()
        log(f"SHADOW OPEN vid={vid} {action} price={current_price:.2f} size={size} state={state}")
        return vid

    def simulate_close(self, vid, current_price, reason="signal"):
        """Close a virtual position and record the trade."""
        if vid not in self.virtual_positions:
            return 0.0
        vp = self.virtual_positions[vid]
        dir_sign = 1 if vp["dir"] == "buy" else -1
        profit = dir_sign * (current_price - vp["entry_price"]) * vp["volume"]
        self.virtual_pnl += profit
        trade_record = {
            "id": vid,
            "dir": vp["dir"],
            "entry_price": vp["entry_price"],
            "exit_price": current_price,
            "volume": vp["volume"],
            "profit": profit,
            "bars": vp["bars"],
            "state": vp["entry_state"],
            "action": vp["action"],
            "entry_time": vp["entry_time"],
            "exit_time": datetime.now().isoformat()[:19],
            "reason": reason,
            "type": "shadow",
            "symbol": SYMBOL
        }
        self.virtual_trades_log.append(trade_record)
        log(f"SHADOW CLOSE vid={vid} {vp['dir']} profit={profit:+.2f} bars={vp['bars']} "
            f"reason={reason} total_pnl={self.virtual_pnl:+.2f}")
        del self.virtual_positions[vid]
        return profit

    def update_virtual_positions(self, current_price, atr, h4_dir, current_state):
        """Update interim rewards for all open virtual positions."""
        total_reward = 0.0
        count = 0
        for vid in list(self.virtual_positions.keys()):
            vp = self.virtual_positions[vid]
            vp["bars"] += 1
            dir_sign = 1 if vp["dir"] == "buy" else -1

            profit = dir_sign * (current_price - vp["entry_price"]) * vp["volume"]
            float_reward = profit / atr if abs(atr) > 1e-10 else 0.0
            aligned = (dir_sign == 1 and h4_dir == 1) or (dir_sign == -1 and h4_dir == -1)
            dir_bonus = 1.0 if aligned else -1.0
            holding_cost = HOLDING_COST * vp["bars"]
            combined = float_reward + dir_bonus + holding_cost

            s_entry = vp["entry_state"]
            a_entry = vp["action"]
            if s_entry not in self.q_table:
                self.q_table[s_entry] = {"buy": 0, "sell": 0, "hold": 0}
            mf = max(self.q_table[current_state].values()) if current_state in self.q_table else 0
            self.q_table[s_entry][a_entry] += self.alpha * (
                combined + self.gamma * mf - self.q_table[s_entry][a_entry]
            )
            total_reward += combined
            count += 1

            log(f"SHADOW INTERIM vid={vid} {vp['dir']} "
                f"flP&L={float_reward:+.3f} dir={dir_bonus:.0f} "
                f"hold={holding_cost:.2f} tot={combined:+.3f} "
                f"bars={vp['bars']} state={s_entry} "
                f"shadow_pnl={self.virtual_pnl:+.2f}")

        if count > 0:
            self.save()
            log(f"SHADOW INTERIM sum_reward={total_reward:+.3f} "
                f"open_pos={len(self.virtual_positions)} "
                f"cum_pnl={self.virtual_pnl:+.2f}")
        return count

    def close_all_virtual(self, current_price, reason="cycle_end"):
        """Close all virtual positions (e.g., at end of cycle)."""
        total = 0.0
        for vid in list(self.virtual_positions.keys()):
            total += self.simulate_close(vid, current_price, reason=reason)
        return total

    def load_persistent_state(self):
        """Load virtual P&L and trade log from file (persists between cron runs)."""
        if os.path.exists(SHADOW_STATE_FILE):
            try:
                with open(SHADOW_STATE_FILE, "r") as f:
                    state = json.load(f)
                self.virtual_pnl = state.get("pnl", 0.0)
                self.virtual_trades_log = state.get("trades", [])
                self.next_id = state.get("next_id", 0)
                # Restore last open position (if any) so next cycle can close it
                last_pos = state.get("last_position")
                if last_pos:
                    vid = last_pos["vid"]
                    self.virtual_positions[vid] = {
                        "dir": last_pos["dir"],
                        "entry_price": last_pos["entry_price"],
                        "volume": last_pos["volume"],
                        "entry_state": tuple(last_pos["entry_state"]),
                        "action": last_pos["action"],
                        "bars": last_pos.get("bars", 0),
                        "entry_time": last_pos.get("entry_time", "")
                    }
                log(f"SHADOW STATE: loaded pnl={self.virtual_pnl:+.2f} "
                    f"trades={len(self.virtual_trades_log)} "
                    f"next_id={self.next_id} "
                    f"restored_pos={vid if last_pos else 'none'}")
            except Exception as e:
                log(f"SHADOW STATE: load error {e}")

    def save_persistent_state(self):
        """Save virtual P&L, trade log, and last open position to file."""
        try:
            trades = self.virtual_trades_log[-500:]
            # Save the most recent open position (if any) for next cycle
            last_position = None
            for vid in self.virtual_positions:
                vp = self.virtual_positions[vid]
                last_position = {
                    "vid": vid,
                    "dir": vp["dir"],
                    "entry_price": vp["entry_price"],
                    "volume": vp["volume"],
                    "entry_state": list(vp["entry_state"]),
                    "action": vp["action"],
                    "bars": vp["bars"],
                    "entry_time": vp.get("entry_time", "")
                }
            state = {
                "pnl": self.virtual_pnl,
                "trades": trades,
                "next_id": self.next_id,
                "last_position": last_position,
                "updated": datetime.now().isoformat()[:19]
            }
            with open(SHADOW_STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            log(f"SHADOW STATE: save error {e}")

    def get_summary(self):
        return {
            "virtual_pnl": round(self.virtual_pnl, 2),
            "open_positions": len(self.virtual_positions),
            "total_trades": len(self.virtual_trades_log),
            "q_table_size": len(self.q_table)
        }


# ---------------------------------------------------------------------------
# Self-Learning: analyze past trades for profitable patterns
# ---------------------------------------------------------------------------

def self_learn(agent, shadow, atr, mtf):
    """Analyze the last 50 trades (real + shadow) and adjust Q-values.

    For each state, calculate the average profit of trades that entered that state.
    Boost Q-values for states with positive avg profit, reduce for negative ones.
    This is offline/batch learning from history, complementary to online Q-learning.
    """
    trades = []

    # 1. Collect real trades from bridge history
    try:
        req = urllib.request.urlopen(BRIDGE + "/history", timeout=5)
        deals = json.loads(req.read().decode()).get("deals", [])
        for d in deals:
            if COMMENT in d.get("comment", "") and d.get("profit", 0) != 0:
                trades.append({
                    "state": None,        # we don't have state for old closed trades
                    "action": d.get("type", "unknown").lower(),
                    "profit": float(d.get("profit", 0)),
                    "type": "live"
                })
    except Exception as e:
        log(f"SELF-LEARN: could not fetch history: {e}")

    # 2. Collect shadow trades from virtual log (include persistent state)
    for t in shadow.virtual_trades_log[-50:]:
        trades.append({
            "state": t.get("state"),
            "action": t.get("action"),
            "profit": t.get("profit", 0),
            "type": "shadow"
        })

    # 3. Keep only last 50
    trades = trades[-50:]

    if len(trades) < 5:
        log(f"SELF-LEARN: only {len(trades)} trades, need at least 5 — skipping")
        return 0

    log(f"SELF-LEARN: analyzing {len(trades)} trades ({sum(1 for t in trades if t['type']=='live')} live, "
        f"{sum(1 for t in trades if t['type']=='shadow')} shadow)")

    # 4. Group by state (only shadow trades have state info)
    state_stats = {}
    for t in trades:
        s = t.get("state")
        if s is None:
            continue
        s = tuple(s) if isinstance(s, list) else s
        if s not in state_stats:
            state_stats[s] = {"profits": [], "count": 0, "actions": {}}
        state_stats[s]["profits"].append(t["profit"])
        state_stats[s]["count"] += 1
        a = t["action"]
        if a not in state_stats[s]["actions"]:
            state_stats[s]["actions"][a] = []
        state_stats[s]["actions"][a].append(t["profit"])

    if not state_stats:
        log("SELF-LEARN: no trades with state info — skipping")
        return 0

    # 5. Adjust Q-values for each state
    adjustments = 0
    for s, stats in state_stats.items():
        if stats["count"] < 2:
            continue
        # Ensure all states in Q-tables are tuples
        s_key = s if isinstance(s, tuple) else tuple(s) if isinstance(s, list) else s
        # Use s_key for Q-table lookups
        s = s_key
        avg_profit = np.mean(stats["profits"])
        # Normalize: scale to [-1, 1] range (cap at ±20)
        norm_profit = np.clip(avg_profit / 20.0, -1.0, 1.0)

        # For each action taken in this state, adjust its Q-value
        for a, profits in stats["actions"].items():
            action_avg = np.mean(profits)
            action_norm = np.clip(action_avg / 20.0, -1.0, 1.0)

            # Boost: alpha * action_norm * 0.5 (conservative offline learning rate)
            if s_key in agent.q_table and a in agent.q_table[s_key]:
                boost = 0.01 * action_norm  # small offline learning rate
                agent.q_table[s][a] += boost

            if s_key in shadow.q_table and a in shadow.q_table[s_key]:
                boost = 0.01 * action_norm
                shadow.q_table[s][a] += boost

            adjustments += 1

    if adjustments > 0:
        agent.save()
        shadow.save()
        log(f"SELF-LEARN: adjusted {adjustments} Q-values across {len(state_stats)} states")

    # 6. Save trades to persistent history file for offline analysis
    try:
        old_trades = []
        if os.path.exists(TRADE_HISTORY):
            with open(TRADE_HISTORY, "r") as f:
                old_trades = json.load(f)
        # Merge: keep last 500
        all_trades = (old_trades + shadow.virtual_trades_log)[-500:]
        # Add live trades from bridge
        for d in deals if 'deals' in dir() else []:
            pass  # skip for now — shadow trades are the primary source
        with open(TRADE_HISTORY, "w") as f:
            json.dump(all_trades, f, indent=2)
    except Exception as e:
        log(f"SELF-LEARN: could not save trade history: {e}")

    return adjustments


# ---------------------------------------------------------------------------
# Swap Mechanism
# ---------------------------------------------------------------------------

def load_swap_state():
    """Load swap state from JSON file (persists between cron runs)."""
    default = {
        "live_pnl": 0.0,
        "shadow_pnl": 0.0,
        "last_swap_time": None,
        "swap_count": 0,
        "last_check_time": None,
        "live_pnl_at_swap": 0.0,
        "shadow_pnl_at_swap": 0.0
    }
    if os.path.exists(SWAP_STATE_FILE):
        try:
            with open(SWAP_STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def save_swap_state(state):
    with open(SWAP_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def compute_live_pnl():
    """Compute cumulative P&L from closed EVA-RL trades via bridge history."""
    total = 0.0
    try:
        req = urllib.request.urlopen(BRIDGE + "/history", timeout=5)
        deals = json.loads(req.read().decode()).get("deals", [])
        for d in deals:
            if COMMENT in d.get("comment", "") and d.get("profit", 0) != 0:
                total += float(d["profit"])
    except Exception as e:
        log(f"SWAP: could not fetch history: {e}")
    return total


def check_swap(live_agent, shadow_agent, atr):
    """Compare live P&L vs shadow P&L. If shadow outperforms by 10%+, swap Q-tables.

    The swap:
    - shadow Q-table becomes the new live Q-table
    - old live Q-table becomes the new shadow Q-table
    - Both are saved to persistent files
    - The swap event is logged
    """
    state = load_swap_state()
    live_pnl = compute_live_pnl()
    shadow_pnl = shadow_agent.virtual_pnl
    now = datetime.now().isoformat()[:19]

    state["live_pnl"] = live_pnl
    state["shadow_pnl"] = shadow_pnl
    state["last_check_time"] = now

    log(f"SWAP CHECK: live_pnl={live_pnl:+.2f} shadow_pnl={shadow_pnl:+.2f} "
        f"diff={shadow_pnl - live_pnl:+.2f}")

    # Only swap if we have enough data (at least 5 trades on shadow)
    if len(shadow_agent.virtual_trades_log) < 5:
        log("SWAP: not enough shadow trades yet — skipping")
        save_swap_state(state)
        return False

    # Check if shadow outperforms live by 10%+
    # Use ratio: if live is near zero, use absolute difference
    if abs(live_pnl) > 0.5:
        outperformance = (shadow_pnl - live_pnl) / abs(live_pnl)
    else:
        outperformance = shadow_pnl - live_pnl  # absolute diff when live is ~0

    outperformance_pct = outperformance * 100 if abs(live_pnl) > 0.5 else outperformance

    log(f"SWAP: outperformance={outperformance_pct:+.1f}% "
        f"(threshold=10% | shadow_trades={len(shadow_agent.virtual_trades_log)})")

    if outperformance > 0.10:  # 10%+ outperformance
        log("SWAP: *** SHADOW OUTPERFORMING LIVE — SWAPPING Q-TABLES ***")
        log(f"SWAP: live_pnl={live_pnl:+.2f} shadow_pnl={shadow_pnl:+.2f} "
            f"outperformance={outperformance_pct:+.1f}%")

        # Perform the swap: exchange Q-tables and save
        live_q = live_agent.q_table
        shadow_q = shadow_agent.q_table

        # Swap
        live_agent.q_table = shadow_q
        shadow_agent.q_table = live_q

        # Save both
        live_agent.save()
        shadow_agent.save()

        # Update swap state
        state["swap_count"] = state.get("swap_count", 0) + 1
        state["last_swap_time"] = now
        state["live_pnl_at_swap"] = live_pnl
        state["shadow_pnl_at_swap"] = shadow_pnl

        # Log to swap log
        log(f"SWAP #{state['swap_count']} completed — "
            f"shadow_pnl={shadow_pnl:+.2f} vs live_pnl={live_pnl:+.2f} "
            f"outperformance={outperformance_pct:+.1f}%",
            f=SWAP_LOG)

        save_swap_state(state)
        return True
    else:
        log("SWAP: no swap needed")
        save_swap_state(state)
        return False


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

log("=" * 60)
log(f"EVA-RL v4 SHADOW + SELF-LEARN + SWAP [{SYMBOL}]")

# ---- 1. FETCH DATA ----
try:
    with urllib.request.urlopen(BRIDGE + "/ohlcv/" + SYMBOL + "/200/M15", timeout=10) as r:
        df = pd.DataFrame(json.loads(r.read().decode())["bars"])
except Exception as e:
    log("ERR data: " + str(e))
    exit()

# Multi-timeframe features
mtf = {}
for tf in ["H4", "H1", "M5"]:
    try:
        with urllib.request.urlopen(BRIDGE + "/ohlcv/" + SYMBOL + "/50/" + tf, timeout=5) as r:
            bars = json.loads(r.read().decode())["bars"]
            prices = [b["close"] for b in bars]
            highs = [b["high"] for b in bars]
            lows = [b["low"] for b in bars]
            if len(prices) > 10:
                ema20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
                mtf[tf.lower() + "_dir"] = 1 if prices[-1] > ema20 else -1
                mtf[tf.lower() + "_rsi"] = 50
                mtf[tf.lower() + "_atr"] = np.mean([h - l for h, l in
                                                     zip(highs[-14:], lows[-14:])]) / prices[-1]
    except Exception:
        pass

try:
    with urllib.request.urlopen(BRIDGE + "/ohlcv/" + SYMBOL + "/10/M5", timeout=5) as r:
        bars5 = json.loads(r.read().decode())["bars"]
        if len(bars5) >= 3:
            mtf["m5_trend"] = 1 if bars5[-1]["close"] > bars5[-3]["close"] else -1
except Exception:
    pass

current_price = float(df["close"].values[-1])
atr = np.mean(df["high"].values[-14:] - df["low"].values[-14:])

# ---- 2. INITIALIZE AGENTS ----
live = RLAgent()
shadow = ShadowAgent()

# Get current state
state = live.get_state(df, mtf)
live.last_state = state
shadow.last_state = state

# ---- 3. LIVE AGENT: TRADE ----
live_action = live.act(state)
live_interim_count = live.update_from_open_positions(mtf, atr, state)

# Check current live positions
try:
    with urllib.request.urlopen(BRIDGE + "/positions", timeout=5) as r:
        pos = json.loads(r.read().decode()).get("positions", [])
        my_pos = [p for p in pos if COMMENT in p.get("comment", "")]
        live_existing = len(my_pos)
except Exception:
    live_existing = 0

live_size = live.get_size(state, live_action)
log(f"LIVE: state={state} act={live_action} pos={live_existing} "
    f"size={live_size} eps={live.epsilon:.3f}")

if live_action in ["buy", "sell"] and live_existing < MAX_POS:
    order = {"symbol": SYMBOL, "volume": live_size, "type": live_action, "comment": COMMENT}
    try:
        req = urllib.request.Request(
            BRIDGE + "/trade",
            data=json.dumps(order).encode(),
            headers={"Content-Type": "application/json"}
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read().decode())
        log(f"LIVE ORDER: {resp}")
    except Exception as e:
        log(f"LIVE ORDER ERR: {e}")

# Live: update from closed trades
try:
    with urllib.request.urlopen(BRIDGE + "/history", timeout=5) as r:
        deals = json.loads(r.read().decode()).get("deals", [])
        for d in deals:
            if COMMENT in d.get("comment", "") and d.get("profit", 0) != 0:
                live.update(state, live_action, float(d["profit"]), state)
                log(f"LIVE CLOSED tkt={d.get('ticket','?')} profit={float(d['profit']):+.2f}")
except Exception:
    pass

# ---- 4. SHADOW AGENT: SIMULATE ----
# Load persistent state (P&L, trade log from previous runs)
shadow.load_persistent_state()

shadow_action = shadow.act(state)

# Close any existing virtual position from previous run at current price
for vid in list(shadow.virtual_positions.keys()):
    shadow.simulate_close(vid, current_price, reason="new_cycle")

# Close any virtual positions that conflict with the new action
# (e.g., was in buy, now wants sell or hold)
for vid in list(shadow.virtual_positions.keys()):
    vp = shadow.virtual_positions[vid]
    if shadow_action == "hold" or (shadow_action == "buy" and vp["dir"] == "sell") or \
       (shadow_action == "sell" and vp["dir"] == "buy"):
        shadow.simulate_close(vid, current_price, reason="signal_change")

# Update interim rewards for remaining virtual positions
shadow.update_virtual_positions(current_price, atr, mtf.get("h4_dir", 0), state)

# Open new shadow trade if action allows
if shadow_action in ["buy", "sell"]:
    shadow.simulate_open(shadow_action, state, current_price, atr)

# Save persistent state
shadow.save_persistent_state()

shadow_summary = shadow.get_summary()
log(f"SHADOW: state={state} act={shadow_action} "
    f"open={shadow_summary['open_positions']} "
    f"total_trades={shadow_summary['total_trades']} "
    f"cum_pnl={shadow_summary['virtual_pnl']:+.2f} "
    f"q_size={shadow_summary['q_table_size']}")

# ---- 5. SELF-LEARNING ----
self_adjustments = self_learn(live, shadow, atr, mtf)
if self_adjustments:
    log(f"SELF-LEARN: adjusted {self_adjustments} Q-values")

# ---- 6. SWAP CHECK ----
swapped = check_swap(live, shadow, atr)
if swapped:
    log("SWAP: Q-tables have been swapped")

log("done")