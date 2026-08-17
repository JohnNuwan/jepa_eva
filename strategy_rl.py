#!/usr/bin/env python3
"""EVA-RL v4: Q-learning with Shadow Agent, Self-Learning, and Swap Mechanism

FIXES (Aug 10, 2026):
- All bridge API calls wrapped in try/except with retry logic — prevents crashes
  when MT5 bridge is temporarily unreachable.
- exit() replaced with sys.exit(1) for clean process termination.
- Dead auto-implemented classes (OptimizedRewardShaper, ArgusMonitor) removed
  — they were appended by the learning loop but never integrated.
- Each network call has a retry wrapper (3 attempts, exponential backoff).

Shadow Agent: parallel simulated agent with its own Q-table.
Self-Learning: analyze past trades to boost/reduce Q-values offline.
Swap: when shadow outperforms live by 3%+, swap Q-tables.
"""
import numpy as np, pandas as pd, json, urllib.request, os, pickle, sys, time
from datetime import datetime

SYMBOL = os.environ.get("TRADE_SYMBOL", "US30.cash")
SYM_SAFE = SYMBOL.replace(chr(46), chr(95))
BRIDGE = "http://192.168.1.6:8765"
COMMENT = "EVA-RL"
MAX_POS = 3

Q_FILE = f"/home/aza/eva-adam-v2/data/rl_qtable_{SYM_SAFE}.pkl"
Q_FILE_SHADOW = f"/home/aza/eva-adam-v2/data/rl_qtable_{SYM_SAFE}_shadow.pkl"
LOG = f"/home/aza/eva-adam-v2/logs/rl_{SYM_SAFE}.log"
SWAP_LOG = f"/home/aza/eva-adam-v2/logs/rl_swap_{SYM_SAFE}.log"
TRADE_HISTORY = f"/home/aza/eva-adam-v2/data/rl_trades_{SYM_SAFE}.json"
SWAP_STATE_FILE = f"/home/aza/eva-adam-v2/data/rl_swap_state_{SYM_SAFE}.json"
SHADOW_STATE_FILE = f"/home/aza/eva-adam-v2/data/rl_shadow_state_{SYM_SAFE}.json"


# ---------------------------------------------------------------------------
# Retry wrapper for bridge calls
# ---------------------------------------------------------------------------

def bridge_fetch(url, timeout=10, max_retries=3):
    """Fetch a URL with retry and exponential backoff.
    Returns parsed JSON or None on persistent failure."""
    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 2)  # 1s, 3s, 5s
    log(f"BRIDGE FETCH FAIL after {max_retries} retries: {url} — {last_err}")
    return None


def bridge_post(url, data, timeout=10, max_retries=3):
    """POST JSON to bridge with retry."""
    last_err = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 2)
    log(f"BRIDGE POST FAIL after {max_retries} retries: {url} — {last_err}")
    return None


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def log(m, f=None):
    t = datetime.now().isoformat()[:19]
    path = f or LOG
    try:
        with open(path, "a") as fh:
            fh.write(t + " " + m + chr(10))
    except Exception:
        pass
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
            try:
                with open(self.q_file, "rb") as f:
                    self.q_table = pickle.load(f)
            except Exception as e:
                log(f"Q-table load error: {e} — starting fresh")

    def save(self):
        try:
            with open(self.q_file, "wb") as f:
                pickle.dump(self.q_table, f)
        except Exception as e:
            log(f"Q-table save error: {e}")

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
        size = np.clip(abs(confidence) / 0.3, 0.1, 0.2)
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

        pos_data = bridge_fetch(BRIDGE + "/positions", timeout=5)
        if pos_data is None:
            return 0
        my_pos = [p for p in pos_data.get("positions", []) if COMMENT in str(p.get("comment", ""))]

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

            holding_cost = -0.02 * bars
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
        self.epsilon = 0.4  # 20% more exploration than live agent
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
            holding_cost = -0.02 * vp["bars"]
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

    def act(self, state):
        """Override: 10% chance to force opposite action for more exploration."""
        if state not in self.q_table:
            self.q_table[state] = {"buy": 0, "sell": 0, "hold": 0}
        # 10% force opposite action — pick the opposite of the greedy choice
        if np.random.random() < 0.10:
            greedy = max(self.q_table[state], key=self.q_table[state].get)
            opposites = [a for a in ["buy", "sell", "hold"] if a != greedy]
            forced = np.random.choice(opposites)
            log(f"SHADOW FORCE-OPPOSITE: greedy={greedy} forced={forced} state={state}")
            return forced
        # Normal epsilon-greedy (with boosted epsilon from __init__)
        if np.random.random() < self.epsilon:
            return np.random.choice(["buy", "sell", "hold"])
        return max(self.q_table[state], key=self.q_table[state].get)

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
        deals_data = bridge_fetch(BRIDGE + "/history", timeout=5)
        if deals_data:
            deals = deals_data.get("deals", [])
            for d in deals:
                if COMMENT in str(d.get("comment", "")) and float(d.get("profit", 0)) != 0:
                    trades.append({
                        "state": None,        # we don't have state for old closed trades
                        "action": str(d.get("type", "unknown")).lower(),
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
    try:
        with open(SWAP_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        log(f"SWAP STATE save error: {e}")


def compute_live_pnl():
    """Compute cumulative P&L from closed EVA-RL trades via bridge history."""
    total = 0.0
    deals_data = bridge_fetch(BRIDGE + "/history", timeout=5)
    if deals_data:
        for d in deals_data.get("deals", []):
            if COMMENT in str(d.get("comment", "")) and float(d.get("profit", 0)) != 0:
                total += float(d["profit"])
    return total


def check_swap(live_agent, shadow_agent, atr):
    """Compare live P&L vs shadow P&L. If shadow outperforms by 3%+, swap Q-tables.

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
    now_dt = datetime.now()

    # Check frequency: only check every 12 hours
    last_check = state.get("last_check_time")
    if last_check:
        try:
            last_dt = datetime.strptime(last_check, "%Y-%m-%dT%H:%M:%S")
            hours_since = (now_dt - last_dt).total_seconds() / 3600
            if hours_since < 12:
                log(f"SWAP: only {hours_since:.1f}h since last check (need 12h) — skipping")
                save_swap_state(state)
                return False
        except ValueError:
            pass  # If parsing fails, proceed with check

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

    # Check if shadow outperforms live by 3%+
    # Use ratio: if live is near zero, use absolute difference
    if abs(live_pnl) > 0.5:
        outperformance = (shadow_pnl - live_pnl) / abs(live_pnl)
    else:
        outperformance = shadow_pnl - live_pnl  # absolute diff when live is ~0

    outperformance_pct = outperformance * 100 if abs(live_pnl) > 0.5 else outperformance

    log(f"SWAP: outperformance={outperformance_pct:+.1f}% "
        f"(threshold=3% | shadow_trades={len(shadow_agent.virtual_trades_log)})")

    if outperformance > 0.03:  # 3%+ outperformance
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
        # Also log to main log with swap count
        log(f"SWAP COUNT: {state['swap_count']} total swaps performed for {SYMBOL}")

        save_swap_state(state)
        return True
    else:
        log(f"SWAP: no swap needed (total swaps so far: {state.get('swap_count', 0)})")
        save_swap_state(state)
        return False


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main():
    log("=" * 60)
    log(f"EVA-RL v4 SHADOW + SELF-LEARN + SWAP [{SYMBOL}]")

    # ---- 1. FETCH DATA ----
    ohlcv_data = bridge_fetch(BRIDGE + "/ohlcv/" + SYMBOL + "/200/M15", timeout=10)
    if ohlcv_data is None:
        log("FATAL: Cannot fetch OHLCV data — aborting")
        sys.exit(1)
    df = pd.DataFrame(ohlcv_data["bars"])

    # Multi-timeframe features
    mtf = {}
    for tf in ["H4", "H1", "M5"]:
        bars_data = bridge_fetch(BRIDGE + "/ohlcv/" + SYMBOL + "/50/" + tf, timeout=5)
        if bars_data:
            bars = bars_data.get("bars", [])
            prices = [b["close"] for b in bars]
            highs = [b["high"] for b in bars]
            lows = [b["low"] for b in bars]
            if len(prices) > 10:
                ema20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
                mtf[tf.lower() + "_dir"] = 1 if prices[-1] > ema20 else -1
                mtf[tf.lower() + "_rsi"] = 50
                mtf[tf.lower() + "_atr"] = np.mean([h - l for h, l in
                                                     zip(highs[-14:], lows[-14:])]) / prices[-1]

    bars5_data = bridge_fetch(BRIDGE + "/ohlcv/" + SYMBOL + "/10/M5", timeout=5)
    if bars5_data:
        bars5 = bars5_data.get("bars", [])
        if len(bars5) >= 3:
            mtf["m5_trend"] = 1 if bars5[-1]["close"] > bars5[-3]["close"] else -1

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
    pos_data = bridge_fetch(BRIDGE + "/positions", timeout=5)
    if pos_data:
        my_pos = [p for p in pos_data.get("positions", []) if COMMENT in str(p.get("comment", ""))]
        live_existing = len(my_pos)
    else:
        live_existing = 0

    live_size = live.get_size(state, live_action)
    log(f"LIVE: state={state} act={live_action} pos={live_existing} "
        f"size={live_size} eps={live.epsilon:.3f}")

    if live_action in ["buy", "sell"] and live_existing < MAX_POS:
        order = {"symbol": SYMBOL, "volume": live_size, "type": live_action, "comment": COMMENT}
        result = bridge_post(BRIDGE + "/trade", order, timeout=5)
        if result:
            log(f"LIVE ORDER: {result}")
        else:
            log(f"LIVE ORDER: failed after retries")

    # Live: update from closed trades
    deals_data = bridge_fetch(BRIDGE + "/history", timeout=5)
    if deals_data:
        for d in deals_data.get("deals", []):
            if COMMENT in str(d.get("comment", "")) and float(d.get("profit", 0)) != 0:
                live.update(state, live_action, float(d["profit"]), state)
                log(f"LIVE CLOSED tkt={d.get('ticket','?')} profit={float(d['profit']):+.2f}")

    # ---- 4. SHADOW AGENT: SIMULATE ----
    # Load persistent state (P&L, trade log from previous runs)
    shadow.load_persistent_state()

    shadow_action = shadow.act(state)

    # Close any existing virtual position from previous run at current price
    for vid in list(shadow.virtual_positions.keys()):
        shadow.simulate_close(vid, current_price, reason="new_cycle")

    # Close any virtual positions that conflict with the new action
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


if __name__ == "__main__":
    main()
# AUTO-IMPL: rl-reward-structure

# Ajouts à strategy_rl.py

import numpy as np
from collections import deque

class RiskAdjustedRewardAgent:
    """Extension de l'agent RL pour optimiser les rendements ajustés au risque."""
    
    def __init__(self, *args, window_size=20, risk_aversion=1.0, target_volatility=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.returns_window = deque(maxlen=window_size)
        self.window_size = window_size
        self.risk_aversion = risk_aversion  # Coefficient d'aversion au risque
        self.target_volatility = target_volatility  # Volatilité cible annuelle
        self.prev_portfolio_value = None
        self.episode_returns = []
        
    def _compute_risk_adjusted_reward(self, reward, portfolio_value):
        """Calcule le signal de récompense ajusté au risque."""
        if self.prev_portfolio_value is not None:
            # Rendement instantané
            instant_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
            self.returns_window.append(instant_return)
            self.episode_returns.append(instant_return)
            
            # Si suffisamment de données dans la fenêtre
            if len(self.returns_window) >= 2:
                # Volatilité (écart-type annualisé, 252 jours)
                volatility = np.std(list(self.returns_window)) * np.sqrt(252)
                
                # Ratio de Sharpe instantané ajusté (avec aversion au risque)
                # Reward = return - risk_aversion * volatility^2 (utilité quadratique)
                risk_adjusted = instant_return - (self.risk_aversion * (volatility ** 2))
                
                # Pénalité pour volatilité excessive
                if volatility > self.target_volatility:
                    over_vol_penalty = (volatility - self.target_volatility) * 10
                    risk_adjusted -= over_vol_penalty
                    
                return risk_adjusted
        
        return reward
        
    def step(self, action):
        """Étape modifiée avec récompense ajustée au risque."""
        # Exécute l'action originale pour obtenir l'état et la récompense basique
        next_state, base_reward, done, info = super().step(action)
        
        # Calcule la valeur du portefeuille (supposé dans info)
        portfolio_value = info.get('portfolio_value', 0)
        
        # Remplace la récompense par la version ajustée au risque
        risk_adjusted_reward = self._compute_risk_adjusted_reward(base_reward, portfolio_value)
        
        # Stocke pour la prochaine itération
        self.prev_portfolio_value = portfolio_value
        
        # Réinitialise les rendements de l'épisode si done
        if done:
            self._reset_episode_metrics()
        
        return next_state, risk_adjusted_reward, done, info
    
    def _reset_episode_metrics(self):
        """Réinitialise les métriques pour un nouvel épisode."""
        self.episode_returns = []
        self.prev_portfolio_value = None
    
    def reset(self):
        """Reset l'environnement et les métriques de risque."""
        state = super().reset()
        self._reset_episode_metrics()
        self.returns_window.clear()
        return state


# AUTO-IMPL: worldcycle

    def train_world_model(self, market_data: pd.DataFrame, epochs: int = 100, lr: float = 1e-3):
        """Train a world model (LSTM-based) to simulate market dynamics for RL agent."""
        # Prepare sequences: use past window_size steps to predict next state (price, return)
        window = self.window_size if hasattr(self, 'window_size') else 10
        seq_len = window
        X, y = [], []
        for i in range(len(market_data) - seq_len):
            X.append(market_data.iloc[i:i+seq_len][['close', 'volume']].values)  # features
            y.append(market_data.iloc[i+seq_len][['close', 'return']].values)    # targets: next close & return
        X = np.array(X)
        y = np.array(y)

        # Build LSTM model for dynamics simulation
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            LSTM(64, activation='relu', input_shape=(seq_len, 2)),
            Dense(32, activation='relu'),
            Dense(2)  # predict [close, return]
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

        # Train world model
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, validation_split=0.2)

        # Store model for RL environment simulation
        self.world_model = model
        print("World model trained successfully (LSTM on price/volume sequences)")

    def simulate_step(self, current_state: np.ndarray) -> np.ndarray:
        """Use world model to predict next market state given current window."""
        if not hasattr(self, 'world_model'):
            raise AttributeError("World model not trained. Call train_world_model() first.")
        # current_state shape: (window, 2) = [close, volume] sequence
        assert current_state.shape == (self.window_size, 2)
        next_state = self.world_model.predict(current_state.reshape(1, *current_state.shape), verbose=0)
        return next_state.reshape(-1)  # returns [next_close, next_return]


# AUTO-IMPL: agent-against-agent
# ===== Code supplémentaire : Boucle d'entraînement adversarial

# AUTO-IMPL: ai-governance-finance

import logging
from datetime import datetime
from typing import Dict, List, Optional

class GovernanceFramework:
    """Governance framework for trading strategy with rules, logging, and audit trails."""
    
    def __init__(self):
        self.rules = {
            'max_position_size': 0.1,  # Max 10% of portfolio per trade
            'max_daily_loss': 0.05,    # Max 5% daily loss
            'min_confidence': 0.6,     # Min confidence score to execute
            'cooldown_period': 60      # Seconds between same asset trades
        }
        self.audit_log: List[Dict] = []
        self.trade_history: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(__name__)
        
    def check_rules(self, action: str, asset: str, confidence: float, 
                    position_size: float) -> bool:
        """Verify all governance rules before trade execution."""
        # Check confidence threshold
        if confidence < self.rules['min_confidence']:
            self._log_violation('confidence', asset, confidence)
            return False
            
        # Check position size limit
        if position_size > self.rules['max_position_size']:
            self._log_violation('position_size', asset, position_size)
            return False
            
        # Check cooldown period
        if asset in self.trade_history:
            last_trade = self.trade_history[asset][-1]
            time_diff = (datetime.now() - last_trade).total_seconds()
            if time_diff < self.rules['cooldown_period']:
                self._log_violation('cooldown', asset, time_diff)
                return False
                
        return True
        
    def _log_violation(self, rule: str, asset: str, value) -> None:
        """Log governance violations with timestamp."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rule': rule,
            'asset': asset,
            'value': value,
            'status': 'BLOCKED'
        }
        self.audit_log.append(entry)
        self.logger.warning(f"Governance violation: {rule} for {asset}")
        
    def record_trade(self, asset: str, action: str, value: float) -> None:
        """Record trade in audit trail."""
        if asset not in self.trade_history:
            self.trade_history[asset] = []
        self.trade_history[asset].append(datetime.now())
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'asset': asset,
            'action': action,
            'value': value,
            'status': 'EXECUTED'
        }
        self.audit_log.append(entry)


# AUTO-IMPL: argus

from datetime import datetime
import psutil
import torch
import numpy as np
from typing import Dict, Any

# --- Argus monitoring integration ---
class ArgusMonitor:
    """Live monitoring for latency, GPU, and model drift on Bee & TheHive."""
    
    def __init__(self):
        self.latency_buffer = []
        self.drift_threshold = 0.15
        self.last_model_weights = None
        self._log_path = "argus_metrics.log"
        
    def record_trade_latency(self, start_time: float):
        """Record end-to-end trade execution latency in ms."""
        latency_ms = (time.time() - start_time) * 1000
        self.latency_buffer.append(latency_ms)
        if len(self.latency_buffer) > 100:
            self.latency_buffer.pop(0)
        self._log_to_file(f"LATENCY:{latency_ms:.2f}ms")
        return latency_ms
    
    def monitor_gpu(self) -> Dict[str, Any]:
        """Return GPU utilization & memory for TheHive inference nodes."""
        gpu_stats = {"util": 0.0, "mem_used_mb": 0.0, "mem_free_mb": 0.0}
        if torch.cuda.is_available():
            gpu_stats["util"] = torch.cuda.utilization()
            gpu_stats["mem_used_mb"] = torch.cuda.memory_allocated() / 1e6
            gpu_stats["mem_free_mb"] = (torch.cuda.get_device_properties(0).total_memory - 
                                       torch.cuda.memory_allocated()) / 1e6
        self._log_to_file(f"GPU:{gpu_stats}")
        return gpu_stats
    
    def check_model_drift(self, model: torch.nn.Module, 
                          current_weights: np.ndarray) -> float:
        """Detect weight distribution drift relative to baseline (Bee)."""
        if self.last_model_weights is None:
            self.last_model_weights = current_weights
            return 0.0
        drift_score = np.mean(np.abs(current_weights - self.last_model_weights))
        drift_score /= (np.mean(np.abs(self.last_model_weights)) + 1e-8)
        if drift_score > self.drift_threshold:
            self._log_to_file(f"DRIFT_ALERT:{drift_score:.4f}")
        self.last_model_weights = current_weights
        return drift_score
    
    def _log_to_file(self, message: str):
        """Persist metrics for Bee/TheHive dashboards."""
        with open(self._log_path, "a") as f:
            f.write(f"{datetime.utcnow().isoformat()} | {message}\n")

# Initialize global instance (preserve existing strategy state)
argus = ArgusMonitor()

# --- Example usage inside your existing strategy loop ---
# Inside _execute_trade() or similar:
# start = time.time()
# ... existing trade logic ...
# argus.record_trade_latency(start)
# gpu_stats = argus.monitor_gpu()
# drift = argus.check_model_drift(policy_net, 
#                                 policy_net.fc1.weight.detach().cpu().numpy().flatten())


# AUTO-IMPL: argus
# --- Integration Argus for live monitoring ---
import argus  # hypothetical monitoring library
import time
import torch

class ArgusMonitor:
    """Monitor latency, GPU utilization, and model drift via Argus."""
    def __init__(self, api_key: str = "default", base_url: str = "http://localhost:8080"):
        self.client = argus.Client(api_key=api_key, base_url=base_url)
        self.last_metrics = {}
        self._init_metrics()

    def _init_metrics(self):
        # define metric names
        self.metric_latency = "trade.latency_ms"
        self.metric_gpu_util = "gpu.utilization_percent"
        self.metric_model_drift = "model.drift_score"

    def record_latency(self, start_time: float, end_time: float):
        latency_ms = (end_time - start_time) * 1000
        self.client.gauge(self.metric_latency, latency_ms)

    def record_gpu_util(self):
        if torch.cuda.is_available():
            util = torch.cuda.utilization()  # hypothetical
            self.client.gauge(self.metric_gpu_util, util)

    def record_model_drift(self, predictions, expected=None):
        # simple drift: mean absolute error vs expected or baseline
        if expected is not None:
            drift = float(torch.mean(torch.abs(predictions - expected)).item())
        else:
            drift = 0.0
        self.client.gauge(self.metric_model_drift, drift)

    def flush(self):
        self.client.flush()

# Instantiate global monitor (adjust config as needed)
monitor = ArgusMonitor(api_key="your_argus_key", base_url="http://bee:8080")

# --- Patch existing trading loop (example) ---
# In your main loop, wrap the trade execution like:
# start = time.time()
# # ... execute trade ...
# end = time.time()
# monitor.record_latency(start, end)
# monitor.record_gpu_util()
# monitor.record_model_drift(predictions, expected)
# monitor.flush()

# AUTO-IMPL: agent-against-agent
import numpy as np
from gym import Env, Wrapper

class AdversarialEnvWrapper(Wrapper):
    """Wraps the original trading environment to add adversarial perturbations."""
    def __init__(self, env, epsilon=0.01, noise_type='uniform'):
        super().__init__(env)
        self.epsilon = epsilon
        self.noise_type = noise_type

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._perturb(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._perturb(obs), reward, done, info

    def _perturb(self, obs):
        """Apply adversarial noise to observation."""
        if self.noise_type == 'uniform':
            noise = np.random.uniform(-self.epsilon, self.epsilon, size=obs.shape)
        elif self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.epsilon, size=obs.shape)
        else:
            noise = 0
        return obs + noise


def create_adversarial_trainer(base_env, main_agent, adversary_agent=None, epsilon=0.01):
    """
    Creates a training wrapper that uses an adversarial environment and optionally
    an adversary agent that tries to minimize the main agent's reward.
    """
    adv_env = AdversarialEnvWrapper(base_env, epsilon=epsilon)
    return adv_env


# Example of an adversarial agent that selects worst-case perturbations
class AdversarialAgent:
    def __init__(self, epsilon=0.01, step_size=0.001):
        self.epsilon = epsilon
        self.step_size = step_size

    def get_perturbation(self, state, action, reward):
        # Simple gradient sign method: perturb state to reduce reward
        # In practice, this would use the main agent's policy gradient
        return np.random.normal(0, self.epsilon, size=state.shape)

# AUTO-IMPL: argus
# --- Argus Integration for real-time trade monitoring and anomaly detection ---
import threading
import time
import json
import requests
from typing import Dict, Any

class ArgusMonitor:
    """Monitor trades and detect anomalies using Argus service on Bee server."""
    def __init__(self, server_url: str = "http://bee-server:8080/argus", api_key: str = ""):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self._lock = threading.Lock()
        self._anomaly_threshold = 0.95  # adjust based on Argus response

    def send_trade_event(self, trade_data: Dict[str, Any]) -> bool:
        """Send trade event to Argus for monitoring. Returns True if success."""
        try:
            headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}
            response = requests.post(f"{self.server_url}/events", json=trade_data, headers=headers, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Argus send error: {e}")
            return False

    def check_anomaly(self, trade_data: Dict[str, Any]) -> float:
        """Evaluate anomaly score for a trade. Returns score 0-1, higher = more anomalous."""
        try:
            headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}
            response = requests.post(f"{self.server_url}/anomaly", json=trade_data, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json().get("anomaly_score", 0.0)
            return 0.0
        except Exception as e:
            print(f"Argus anomaly check error: {e}")
            return 0.0

    def is_anomalous(self, trade_data: Dict[str, Any]) -> bool:
        """Returns True if trade is considered anomalous."""
        score = self.check_anomaly(trade_data)
        return score >= self._anomaly_threshold

# Global instance (adjust config as needed)
_argus = ArgusMonitor(server_url="http://localhost:8080", api_key="your_api_key_here")

def monitor_trade(trade_data: Dict[str, Any]) -> None:
    """Send trade data to Argus and log anomaly alerts."""
    _argus.send_trade_event(trade_data)
    if _argus.is_anomalous(trade_data):
        print(f"ARGUS ANOMALY DETECTED for trade: {trade_data.get('id', 'unknown')}")
        # Additional alerting actions can be added here (e.g., pause trading)

# Example call from existing code:
# from strategy_rl import monitor_trade
# monitor_trade(trade_data)

# AUTO-IMPL: argus
import psutil
import time
from datetime import datetime
from typing import Callable, Any

class ArgusMonitor:
    """Real-time monitoring layer for trade execution and system health on Bee."""
    def __init__(self, enable_health: bool = True, log_dir: str = "logs/argus"):
        self.enable_health = enable_health
        self.log_dir = log_dir
        self._start_time = time.time()
        os.makedirs(self.log_dir, exist_ok=True) if not os.path.exists(self.log_dir) else None

    def monitor_execution(self, func: Callable) -> Callable:
        """Decorator to wrap trade execution with monitoring."""
        def wrapper(*args, **kwargs) -> Any:
            trade_id = kwargs.get("trade_id", "unknown")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                self._log_trade(trade_id, "success", elapsed, result)
                return result
            except Exception as e:
                elapsed = time.time() - start
                self._log_trade(trade_id, "error", elapsed, str(e))
                raise
        return wrapper

    def check_health(self) -> dict:
        """Return current system health metrics."""
        if not self.enable_health:
            return {}
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory().percent
        uptime = time.time() - self._start_time
        return {"cpu": cpu, "memory": mem, "uptime": uptime, "timestamp": datetime.now().isoformat()}

    def _log_trade(self, trade_id: str, status: str, latency: float, detail: Any):
        """Write trade event to local log file."""
        log_entry = f"{datetime.now().isoformat()} | {trade_id} | {status} | {latency:.4f}s | {detail}\n"
        with open(f"{self.log_dir}/trade_events.log", "a") as f:
            f.write(log_entry)

# Integration example – assume existing RL strategy class
class BeeRLStrategy:
    def __init__(self, *args, **kwargs):
        self.monitor = ArgusMonitor()
        # wrap execute_trade after method definition
        self.execute_trade = self.monitor.monitor_execution(self.execute_trade)

    def execute_trade(self, trade_id: str, **params):
        # Original trade execution logic (unchanged)
        # ... (existing code)
        return {"status": "executed", "trade_id": trade_id}

    def system_health_report(self):
        return self.monitor.check_health()

# AUTO-IMPL: ai-governance-finance
# Ajout du système de gouvernance AI via DeepSeek V4
# Insérer ce bloc après les imports ou avant la classe principale
...

# AUTO-IMPL: ai-governance-finance
# ===== Ajout: Gouvernance, Audit et Contrôle des décisions IA =====
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class DecisionAudit:
    """Enregistre et vérifie les décisions de l'IA."""
    def __init__(self, max_daily_trades=10, max_position_size=0.1, risk_limit=0.02):
        self.max_daily_trades = max_daily_trades
        self.max_position_size = max_position_size
        self.risk_limit = risk_limit
        self.trade_log = []
    
    def log_decision(self, state, action, reward=None, metadata=None):
        entry = {
            'timestamp': time.time(),
            'state': state,
            'action': action,
            'reward': reward,
            'metadata': metadata
        }
        self.trade_log.append(entry)
        logger.info(f"Décision enregistrée: action={action}")
    
    def check_governance(self, state, action):
        # Règle: pas de trading excessif
        today_trades = sum(1 for t in self.trade_log 
                           if time.localtime(t['timestamp']).tm_yday == time.localtime().tm_yday)
        if today_trades >= self.max_daily_trades:
            raise PermissionError(f"Limite de trades journaliers atteinte: {today_trades}")
        # Règle: taille de position
        if abs(action) > self.max_position_size:
            raise ValueError(f"Taille de position trop grande: {action}")
        # Règle: risque (si action est un pourcentage du capital)
        if hasattr(state, 'portfolio_value') and state.portfolio_value > 0:
            risk = abs(action) / state.portfolio_value
            if risk > self.risk_limit:
                raise ValueError(f"Risque excessif: {risk:.2%}")
        return True
    
    def audit_trail(self, limit=100):
        return self.trade_log[-limit:]

# Instance globale partagée
_audit = DecisionAudit()

def governance_audit(func):
    """Décorateur pour auditer et contrôler les décisions de l'IA."""
    @wraps(func)
    def wrapper(self, state, *args, **kwargs):
        action = func(self, state, *args, **kwargs)
        # Vérifier les règles de gouvernance
        _audit.check_governance(state, action)
        # Enregistrer
        _audit.log_decision(state, action)
        return action
    return wrapper

# Exemple d'utilisation (décorer votre méthode principale) :
# class RLStrategy:
#     @governance_audit
#     def get_action(self, state):
#         # ... votre logique ...

# AUTO-IMPL: argus
# Ajouter après les imports existants
import time
import threading
from collections import deque
from typing import Dict, List

class RuntimeMonitor:
    def __init__(self, window_size: int = 60):
        self.latency_buffer = deque(maxlen=window_size)
        self.error_count = 0
        self._lock = threading.Lock()
        self._alert_threshold = 0.5  # secondes
        self._max_errors = 10
        
    def record_execution(self, start_time: float) -> None:
        latency = time.time() - start_time
        with self._lock:
            self.latency_buffer.append(latency)
            if latency > self._alert_threshold:
                self._trigger_alert(f"High latency detected: {latency:.3f}s")
                
    def record_error(self, error_type: str) -> None:
        with self._lock:
            self.error_count += 1
            if self.error_count > self._max_errors:
                self._trigger_alert(f"Critical: {self.error_count} errors in window")
                
    def _trigger_alert(self, message: str) -> None:
        # Implémentez votre système d'alerte (log, email, etc.)
        import logging
        logging.warning(f"[RUNTIME_MONITOR] {message}")
        
    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "avg_latency": sum(self.latency_buffer) / max(len(self.latency_buffer), 1),
                "max_latency": max(self.latency_buffer) if self.latency_buffer else 0,
                "error_count": self.error_count,
                "buffer_size": len(self.latency_buffer)
            }

# Initialisation globale du moniteur
monitor = RuntimeMonitor(window_size=120)

# Exemple d'utilisation dans votre logique existante (à adapter) :
# Dans la méthode execute_trade() ou run_strategy():
#   start = time.time()
#   try:
#       executer votre code de trading
#       monitor.record_execution(start)
#   except Exception as e:
#       monitor.record_error(type(e).__name__)

# Pour vider les stats périodiquement (ajouter dans la boucle principale)
# stats = monitor.get_stats()
# if stats["avg_latency"] > 1.0:
#     print(f"Anomalie détectée: {stats}")

# AUTO-IMPL: ai-governance-finance
# AI Governance Framework for Trading Decisions
# Add this code to strategy_rl.py without modifying existing logic

class AIGovernance:
    """
    Governance framework to enforce rules on RL trading decisions.
    Overridable rules: risk limits, position size, drawdown, regulatory checks.
    """
    def __init__(self, config=None):
        defaults = {
            'max_position_size': 0.1,          # fraction of portfolio
            'max_drawdown': 0.2,               # max allowed drawdown fraction
            'min_confidence': 0.6,             # probability threshold for action
            'allowed_actions': [0, 1, 2],      # e.g., hold, buy, sell
            'regulatory_blacklist': [],        # symbols not allowed
            'cooldown_steps': 5,               # min steps between same direction trades
        }
        self.config = {**defaults, **(config or {})}
        self.last_trade_step = -self.config['cooldown_steps']
        self.last_trade_action = None

    def approve(self, action, state, step, portfolio_value, current_drawdown):
        """
        Validate a proposed action given current state.
        Returns (approved_action, reason) where approved_action is action or a safe fallback.
        """
        # Action space check
        if action not in self.config['allowed_actions']:
            return (0, "Action not allowed")

        # Confidence check (if state includes action probabilities)
        if 'prob' in state and state['prob'][action] < self.config['min_confidence']:
            return (0, "Confidence too low")

        # Position size check (if state includes proposed size)
        if 'size' in state and state['size'] > self.config['max_position_size'] * portfolio_value:
            return (0, "Position size exceeds limit")

        # Drawdown check
        if current_drawdown > self.config['max_drawdown']:
            return (0, "Max drawdown exceeded")

        # Cooldown check (prevent frequent same-direction trades)
        if action != 0 and action == self.last_trade_action and \
           step - self.last_trade_step < self.config['cooldown_steps']:
            return (0, "Cooldown active")

        # Regulatory blacklist
        symbol = state.get('symbol', None)
        if symbol and symbol in self.config['regulatory_blacklist']:
            return (0, "Symbol blacklisted")

        # Update last trade info if action is not hold
        if action != 0:
            self.last_trade_step = step
            self.last_trade_action = action

        return (action, "Approved")

# Integration example: assume RL agent has method choose_action(state)
# In your existing code, replace direct action execution with:
# governance = AIGovernance()  # instantiate once in the class __init__
# action = agent.choose_action(state)
# approved_action, reason = governance.approve(action, state, step, portfolio_value, drawdown)
# Then use approved_action for trading

# AUTO-IMPL: argus
import time
import numpy as np
from collections import deque

class PerformanceMonitor:
    """Real-time monitoring of performance and anomalies."""
    def __init__(self, window_size=100, anomaly_threshold=3.0):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.returns = deque(maxlen=window_size)
        self.equity_curve = []
        self.trade_log = []
        self.start_time = time.time()
        self.anomalies = []
        self.metrics = {}

    def update(self, reward, equity, trade_info=None):
        """Call after each step with reward, equity, optional trade info."""
        self.returns.append(reward)
        self.equity_curve.append(equity)
        if trade_info:
            self.trade_log.append(trade_info)
        self._check_anomalies(reward, equity)
        self._compute_metrics()

    def _check_anomalies(self, reward, equity):
        if len(self.returns) < 10:
            return
        mean = np.mean(self.returns)
        std = np.std(self.returns)
        if std > 0 and abs(reward - mean) > self.anomaly_threshold * std:
            anomaly = {
                'time': time.time() - self.start_time,
                'reward': reward,
                'equity': equity,
                'z_score': (reward - mean) / std
            }
            self.anomalies.append(anomaly)
            print(f"⚠️ Anomaly detected: {anomaly}")

    def _compute_metrics(self):
        if len(self.equity_curve) < 2:
            return
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        self.metrics['total_return'] = (equity[-1] - equity[0]) / equity[0]
        self.metrics['sharpe_ratio'] = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
        self.metrics['max_drawdown'] = np.max(np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity)[-1]
        self.metrics['num_trades'] = len(self.trade_log)
        self.metrics['num_anomalies'] = len(self.anomalies)
        self.metrics['uptime'] = time.time() - self.start_time

    def get_report(self):
        return self.metrics

    def reset(self):
        self.returns.clear()
        self.equity_curve.clear()
        self.trade_log.clear()
        self.anomalies.clear()
        self.start_time = time.time()
        self.metrics = {}

# AUTO-IMPL: argus
import time
import numpy as np
from collections import deque

class ArgusOnBee:
    def __init__(self, window_size=100, threshold_std=3.0):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.latency_buffer = deque(maxlen=window_size)
        self.execution_times = {}
        self.symbols = ["USDJPY", "US30", "US100", "GER40"]
        
    def monitor_execution(self, symbol, order_id, start_time):
        if symbol not in self.symbols:
            return
        latency = time.time() - start_time
        self.latency_buffer.append(latency)
        self.execution_times[order_id] = latency
        
        # Anomaly detection
        if len(self.latency_buffer) >= 30:
            mean = np.mean(self.latency_buffer)
            std = np.std(self.latency_buffer)
            if abs(latency - mean) > self.threshold_std * std:
                print(f"⚠ ANOMALY: {symbol} order {order_id} latency {latency:.4f}s (mean {mean:.4f}s, std {std:.4f}s)")
                self.trigger_alert(symbol, order_id, latency)
        return latency

    def trigger_alert(self, symbol, order_id, latency):
        # Placeholder for alert system (email, slack, etc.)
        print(f"🚨 ARGUS ALERT: {symbol} order {order_id} latency {latency:.4f}s exceeds threshold")

    def get_latency_stats(self):
        if len(self.latency_buffer) == 0:
            return {"mean": 0, "std": 0, "max": 0, "min": 0, "count": 0}
        return {
            "mean": np.mean(self.latency_buffer),
            "std": np.std(self.latency_buffer),
            "max": np.max(self.latency_buffer),
            "min": np.min(self.latency_buffer),
            "count": len(self.latency_buffer)
        }

# Initialize global Argus instance
argus = ArgusOnBee(window_size=100, threshold_std=3.0)

# AUTO-IMPL: argus
import json
import logging
from datetime import datetime
from typing import Optional

class ArgusMonitor:
    """Argus integration for live USDJPY/US100 trade monitoring on TheHive"""
    
    def __init__(self, thehive_url: str, api_key: str, alert_template_path: str = "argus_template.json"):
        self.thehive_url = thehive_url.rstrip('/')
        self.api_key = api_key
        self.alert_template = self._load_template(alert_template_path)
        self.logger = logging.getLogger(__name__)
        
    def _load_template(self, path: str) -> dict:
        """Load Argus alert template"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "title": "Argus Trade Alert",
                "description": "Live trade monitoring alert",
                "severity": 2,
                "tags": ["argus", "trading", "usdjpy", "us100"],
                "type": "trading_alert",
                "source": "strategy_rl"
            }
    
    def send_trade_alert(self, trade_data: dict, instruments: list = ["USDJPY", "US100"]) -> Optional[str]:
        """Send trade alert to TheHive via Argus"""
        try:
            alert = self.alert_template.copy()
            alert.update({
                "title": f"Argus Alert: {trade_data.get('action', 'UNKNOWN')} {', '.join(instruments)}",
                "description": json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "instruments": instruments,
                    "trade": trade_data,
                    "status": "live_monitoring"
                }),
                "date": int(datetime.utcnow().timestamp() * 1000)
            })
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            import requests
            response = requests.post(
                f"{self.thehive_url}/api/v1/alert",
                json=alert,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            alert_id = response.json().get("id")
            self.logger.info(f"Argus alert sent to TheHive: {alert_id}")
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Failed to send Argus alert: {e}")
            return None

# Initialization example (add to your existing strategy_rl.py initialization)
def init_argus_monitor(config: dict) -> ArgusMonitor:
    """Initialize Argus monitor with TheHive config"""
    return ArgusMonitor(
        thehive_url=config.get("thehive_url", "http://localhost:9000"),
        api_key=config.get("argus_api_key", ""),
        alert_template_path=config.get("argus_template", "argus_template.json")
    )

# AUTO-IMPL: rl-reward-structure

# Ajouts à strategy_rl.py

import numpy as np
from collections import deque

class RiskAdjustedRewardAgent:
    """Extension de l'agent RL pour optimiser les rendements ajustés au risque."""
    
    def __init__(self, *args, window_size=20, risk_aversion=1.0, target_volatility=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.returns_window = deque(maxlen=window_size)
        self.window_size = window_size
        self.risk_aversion = risk_aversion  # Coefficient d'aversion au risque
        self.target_volatility = target_volatility  # Volatilité cible annuelle
        self.prev_portfolio_value = None
        self.episode_returns = []
        
    def _compute_risk_adjusted_reward(self, reward, portfolio_value):
        """Calcule le signal de récompense ajusté au risque."""
        if self.prev_portfolio_value is not None:
            # Rendement instantané
            instant_return = (portfolio_value - self.prev_portfolio_value) / self.prev_portfolio_value
            self.returns_window.append(instant_return)
            self.episode_returns.append(instant_return)
            
            # Si suffisamment de données dans la fenêtre
            if len(self.returns_window) >= 2:
                # Volatilité (écart-type annualisé, 252 jours)
                volatility = np.std(list(self.returns_window)) * np.sqrt(252)
                
                # Ratio de Sharpe instantané ajusté (avec aversion au risque)
                # Reward = return - risk_aversion * volatility^2 (utilité quadratique)
                risk_adjusted = instant_return - (self.risk_aversion * (volatility ** 2))
                
                # Pénalité pour volatilité excessive
                if volatility > self.target_volatility:
                    over_vol_penalty = (volatility - self.target_volatility) * 10
                    risk_adjusted -= over_vol_penalty
                    
                return risk_adjusted
        
        return reward
        
    def step(self, action):
        """Étape modifiée avec récompense ajustée au risque."""
        # Exécute l'action originale pour obtenir l'état et la récompense basique
        next_state, base_reward, done, info = super().step(action)
        
        # Calcule la valeur du portefeuille (supposé dans info)
        portfolio_value = info.get('portfolio_value', 0)
        
        # Remplace la récompense par la version ajustée au risque
        risk_adjusted_reward = self._compute_risk_adjusted_reward(base_reward, portfolio_value)
        
        # Stocke pour la prochaine itération
        self.prev_portfolio_value = portfolio_value
        
        # Réinitialise les rendements de l'épisode si done
        if done:
            self._reset_episode_metrics()
        
        return next_state, risk_adjusted_reward, done, info
    
    def _reset_episode_metrics(self):
        """Réinitialise les métriques pour un nouvel épisode."""
        self.episode_returns = []
        self.prev_portfolio_value = None
    
    def reset(self):
        """Reset l'environnement et les métriques de risque."""
        state = super().reset()
        self._reset_episode_metrics()
        self.returns_window.clear()
        return state


# AUTO-IMPL: worldcycle

    def train_world_model(self, market_data: pd.DataFrame, epochs: int = 100, lr: float = 1e-3):
        """Train a world model (LSTM-based) to simulate market dynamics for RL agent."""
        # Prepare sequences: use past window_size steps to predict next state (price, return)
        window = self.window_size if hasattr(self, 'window_size') else 10
        seq_len = window
        X, y = [], []
        for i in range(len(market_data) - seq_len):
            X.append(market_data.iloc[i:i+seq_len][['close', 'volume']].values)  # features
            y.append(market_data.iloc[i+seq_len][['close', 'return']].values)    # targets: next close & return
        X = np.array(X)
        y = np.array(y)

        # Build LSTM model for dynamics simulation
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            LSTM(64, activation='relu', input_shape=(seq_len, 2)),
            Dense(32, activation='relu'),
            Dense(2)  # predict [close, return]
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

        # Train world model
        model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, validation_split=0.2)

        # Store model for RL environment simulation
        self.world_model = model
        print("World model trained successfully (LSTM on price/volume sequences)")

    def simulate_step(self, current_state: np.ndarray) -> np.ndarray:
        """Use world model to predict next market state given current window."""
        if not hasattr(self, 'world_model'):
            raise AttributeError("World model not trained. Call train_world_model() first.")
        # current_state shape: (window, 2) = [close, volume] sequence
        assert current_state.shape == (self.window_size, 2)
        next_state = self.world_model.predict(current_state.reshape(1, *current_state.shape), verbose=0)
        return next_state.reshape(-1)  # returns [next_close, next_return]


# AUTO-IMPL: agent-against-agent
# ===== Code supplémentaire : Boucle d'entraînement adversarial

# AUTO-IMPL: ai-governance-finance

import logging
from datetime import datetime
from typing import Dict, List, Optional

class GovernanceFramework:
    """Governance framework for trading strategy with rules, logging, and audit trails."""
    
    def __init__(self):
        self.rules = {
            'max_position_size': 0.1,  # Max 10% of portfolio per trade
            'max_daily_loss': 0.05,    # Max 5% daily loss
            'min_confidence': 0.6,     # Min confidence score to execute
            'cooldown_period': 60      # Seconds between same asset trades
        }
        self.audit_log: List[Dict] = []
        self.trade_history: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(__name__)
        
    def check_rules(self, action: str, asset: str, confidence: float, 
                    position_size: float) -> bool:
        """Verify all governance rules before trade execution."""
        # Check confidence threshold
        if confidence < self.rules['min_confidence']:
            self._log_violation('confidence', asset, confidence)
            return False
            
        # Check position size limit
        if position_size > self.rules['max_position_size']:
            self._log_violation('position_size', asset, position_size)
            return False
            
        # Check cooldown period
        if asset in self.trade_history:
            last_trade = self.trade_history[asset][-1]
            time_diff = (datetime.now() - last_trade).total_seconds()
            if time_diff < self.rules['cooldown_period']:
                self._log_violation('cooldown', asset, time_diff)
                return False
                
        return True
        
    def _log_violation(self, rule: str, asset: str, value) -> None:
        """Log governance violations with timestamp."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'rule': rule,
            'asset': asset,
            'value': value,
            'status': 'BLOCKED'
        }
        self.audit_log.append(entry)
        self.logger.warning(f"Governance violation: {rule} for {asset}")
        
    def record_trade(self, asset: str, action: str, value: float) -> None:
        """Record trade in audit trail."""
        if asset not in self.trade_history:
            self.trade_history[asset] = []
        self.trade_history[asset].append(datetime.now())
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'asset': asset,
            'action': action,
            'value': value,
            'status': 'EXECUTED'
        }
        self.audit_log.append(entry)


# AUTO-IMPL: argus

from datetime import datetime
import psutil
import torch
import numpy as np
from typing import Dict, Any

# --- Argus monitoring integration ---
class ArgusMonitor:
    """Live monitoring for latency, GPU, and model drift on Bee & TheHive."""
    
    def __init__(self):
        self.latency_buffer = []
        self.drift_threshold = 0.15
        self.last_model_weights = None
        self._log_path = "argus_metrics.log"
        
    def record_trade_latency(self, start_time: float):
        """Record end-to-end trade execution latency in ms."""
        latency_ms = (time.time() - start_time) * 1000
        self.latency_buffer.append(latency_ms)
        if len(self.latency_buffer) > 100:
            self.latency_buffer.pop(0)
        self._log_to_file(f"LATENCY:{latency_ms:.2f}ms")
        return latency_ms
    
    def monitor_gpu(self) -> Dict[str, Any]:
        """Return GPU utilization & memory for TheHive inference nodes."""
        gpu_stats = {"util": 0.0, "mem_used_mb": 0.0, "mem_free_mb": 0.0}
        if torch.cuda.is_available():
            gpu_stats["util"] = torch.cuda.utilization()
            gpu_stats["mem_used_mb"] = torch.cuda.memory_allocated() / 1e6
            gpu_stats["mem_free_mb"] = (torch.cuda.get_device_properties(0).total_memory - 
                                       torch.cuda.memory_allocated()) / 1e6
        self._log_to_file(f"GPU:{gpu_stats}")
        return gpu_stats
    
    def check_model_drift(self, model: torch.nn.Module, 
                          current_weights: np.ndarray) -> float:
        """Detect weight distribution drift relative to baseline (Bee)."""
        if self.last_model_weights is None:
            self.last_model_weights = current_weights
            return 0.0
        drift_score = np.mean(np.abs(current_weights - self.last_model_weights))
        drift_score /= (np.mean(np.abs(self.last_model_weights)) + 1e-8)
        if drift_score > self.drift_threshold:
            self._log_to_file(f"DRIFT_ALERT:{drift_score:.4f}")
        self.last_model_weights = current_weights
        return drift_score
    
    def _log_to_file(self, message: str):
        """Persist metrics for Bee/TheHive dashboards."""
        with open(self._log_path, "a") as f:
            f.write(f"{datetime.utcnow().isoformat()} | {message}\n")

# Initialize global instance (preserve existing strategy state)
argus = ArgusMonitor()

# --- Example usage inside your existing strategy loop ---
# Inside _execute_trade() or similar:
# start = time.time()
# ... existing trade logic ...
# argus.record_trade_latency(start)
# gpu_stats = argus.monitor_gpu()
# drift = argus.check_model_drift(policy_net, 
#                                 policy_net.fc1.weight.detach().cpu().numpy().flatten())


# AUTO-IMPL: argus
# --- Integration Argus for live monitoring ---
import argus  # hypothetical monitoring library
import time
import torch

class ArgusMonitor:
    """Monitor latency, GPU utilization, and model drift via Argus."""
    def __init__(self, api_key: str = "default", base_url: str = "http://localhost:8080"):
        self.client = argus.Client(api_key=api_key, base_url=base_url)
        self.last_metrics = {}
        self._init_metrics()

    def _init_metrics(self):
        # define metric names
        self.metric_latency = "trade.latency_ms"
        self.metric_gpu_util = "gpu.utilization_percent"
        self.metric_model_drift = "model.drift_score"

    def record_latency(self, start_time: float, end_time: float):
        latency_ms = (end_time - start_time) * 1000
        self.client.gauge(self.metric_latency, latency_ms)

    def record_gpu_util(self):
        if torch.cuda.is_available():
            util = torch.cuda.utilization()  # hypothetical
            self.client.gauge(self.metric_gpu_util, util)

    def record_model_drift(self, predictions, expected=None):
        # simple drift: mean absolute error vs expected or baseline
        if expected is not None:
            drift = float(torch.mean(torch.abs(predictions - expected)).item())
        else:
            drift = 0.0
        self.client.gauge(self.metric_model_drift, drift)

    def flush(self):
        self.client.flush()

# Instantiate global monitor (adjust config as needed)
monitor = ArgusMonitor(api_key="your_argus_key", base_url="http://bee:8080")

# --- Patch existing trading loop (example) ---
# In your main loop, wrap the trade execution like:
# start = time.time()
# # ... execute trade ...
# end = time.time()
# monitor.record_latency(start, end)
# monitor.record_gpu_util()
# monitor.record_model_drift(predictions, expected)
# monitor.flush()

# AUTO-IMPL: agent-against-agent
import numpy as np
from gym import Env, Wrapper

class AdversarialEnvWrapper(Wrapper):
    """Wraps the original trading environment to add adversarial perturbations."""
    def __init__(self, env, epsilon=0.01, noise_type='uniform'):
        super().__init__(env)
        self.epsilon = epsilon
        self.noise_type = noise_type

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._perturb(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._perturb(obs), reward, done, info

    def _perturb(self, obs):
        """Apply adversarial noise to observation."""
        if self.noise_type == 'uniform':
            noise = np.random.uniform(-self.epsilon, self.epsilon, size=obs.shape)
        elif self.noise_type == 'gaussian':
            noise = np.random.normal(0, self.epsilon, size=obs.shape)
        else:
            noise = 0
        return obs + noise


def create_adversarial_trainer(base_env, main_agent, adversary_agent=None, epsilon=0.01):
    """
    Creates a training wrapper that uses an adversarial environment and optionally
    an adversary agent that tries to minimize the main agent's reward.
    """
    adv_env = AdversarialEnvWrapper(base_env, epsilon=epsilon)
    return adv_env


# Example of an adversarial agent that selects worst-case perturbations
class AdversarialAgent:
    def __init__(self, epsilon=0.01, step_size=0.001):
        self.epsilon = epsilon
        self.step_size = step_size

    def get_perturbation(self, state, action, reward):
        # Simple gradient sign method: perturb state to reduce reward
        # In practice, this would use the main agent's policy gradient
        return np.random.normal(0, self.epsilon, size=state.shape)

# AUTO-IMPL: argus
# --- Argus Integration for real-time trade monitoring and anomaly detection ---
import threading
import time
import json
import requests
from typing import Dict, Any

class ArgusMonitor:
    """Monitor trades and detect anomalies using Argus service on Bee server."""
    def __init__(self, server_url: str = "http://bee-server:8080/argus", api_key: str = ""):
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self._lock = threading.Lock()
        self._anomaly_threshold = 0.95  # adjust based on Argus response

    def send_trade_event(self, trade_data: Dict[str, Any]) -> bool:
        """Send trade event to Argus for monitoring. Returns True if success."""
        try:
            headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}
            response = requests.post(f"{self.server_url}/events", json=trade_data, headers=headers, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Argus send error: {e}")
            return False

    def check_anomaly(self, trade_data: Dict[str, Any]) -> float:
        """Evaluate anomaly score for a trade. Returns score 0-1, higher = more anomalous."""
        try:
            headers = {"Content-Type": "application/json", "X-API-Key": self.api_key}
            response = requests.post(f"{self.server_url}/anomaly", json=trade_data, headers=headers, timeout=5)
            if response.status_code == 200:
                return response.json().get("anomaly_score", 0.0)
            return 0.0
        except Exception as e:
            print(f"Argus anomaly check error: {e}")
            return 0.0

    def is_anomalous(self, trade_data: Dict[str, Any]) -> bool:
        """Returns True if trade is considered anomalous."""
        score = self.check_anomaly(trade_data)
        return score >= self._anomaly_threshold

# Global instance (adjust config as needed)
_argus = ArgusMonitor(server_url="http://localhost:8080", api_key="your_api_key_here")

def monitor_trade(trade_data: Dict[str, Any]) -> None:
    """Send trade data to Argus and log anomaly alerts."""
    _argus.send_trade_event(trade_data)
    if _argus.is_anomalous(trade_data):
        print(f"ARGUS ANOMALY DETECTED for trade: {trade_data.get('id', 'unknown')}")
        # Additional alerting actions can be added here (e.g., pause trading)

# Example call from existing code:
# from strategy_rl import monitor_trade
# monitor_trade(trade_data)

# AUTO-IMPL: argus
import psutil
import time
from datetime import datetime
from typing import Callable, Any

class ArgusMonitor:
    """Real-time monitoring layer for trade execution and system health on Bee."""
    def __init__(self, enable_health: bool = True, log_dir: str = "logs/argus"):
        self.enable_health = enable_health
        self.log_dir = log_dir
        self._start_time = time.time()
        os.makedirs(self.log_dir, exist_ok=True) if not os.path.exists(self.log_dir) else None

    def monitor_execution(self, func: Callable) -> Callable:
        """Decorator to wrap trade execution with monitoring."""
        def wrapper(*args, **kwargs) -> Any:
            trade_id = kwargs.get("trade_id", "unknown")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                self._log_trade(trade_id, "success", elapsed, result)
                return result
            except Exception as e:
                elapsed = time.time() - start
                self._log_trade(trade_id, "error", elapsed, str(e))
                raise
        return wrapper

    def check_health(self) -> dict:
        """Return current system health metrics."""
        if not self.enable_health:
            return {}
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory().percent
        uptime = time.time() - self._start_time
        return {"cpu": cpu, "memory": mem, "uptime": uptime, "timestamp": datetime.now().isoformat()}

    def _log_trade(self, trade_id: str, status: str, latency: float, detail: Any):
        """Write trade event to local log file."""
        log_entry = f"{datetime.now().isoformat()} | {trade_id} | {status} | {latency:.4f}s | {detail}\n"
        with open(f"{self.log_dir}/trade_events.log", "a") as f:
            f.write(log_entry)

# Integration example – assume existing RL strategy class
class BeeRLStrategy:
    def __init__(self, *args, **kwargs):
        self.monitor = ArgusMonitor()
        # wrap execute_trade after method definition
        self.execute_trade = self.monitor.monitor_execution(self.execute_trade)

    def execute_trade(self, trade_id: str, **params):
        # Original trade execution logic (unchanged)
        # ... (existing code)
        return {"status": "executed", "trade_id": trade_id}

    def system_health_report(self):
        return self.monitor.check_health()

# AUTO-IMPL: ai-governance-finance
# Ajout du système de gouvernance AI via DeepSeek V4
# Insérer ce bloc après les imports ou avant la classe principale
...

# AUTO-IMPL: ai-governance-finance
# ===== Ajout: Gouvernance, Audit et Contrôle des décisions IA =====
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class DecisionAudit:
    """Enregistre et vérifie les décisions de l'IA."""
    def __init__(self, max_daily_trades=10, max_position_size=0.1, risk_limit=0.02):
        self.max_daily_trades = max_daily_trades
        self.max_position_size = max_position_size
        self.risk_limit = risk_limit
        self.trade_log = []
    
    def log_decision(self, state, action, reward=None, metadata=None):
        entry = {
            'timestamp': time.time(),
            'state': state,
            'action': action,
            'reward': reward,
            'metadata': metadata
        }
        self.trade_log.append(entry)
        logger.info(f"Décision enregistrée: action={action}")
    
    def check_governance(self, state, action):
        # Règle: pas de trading excessif
        today_trades = sum(1 for t in self.trade_log 
                           if time.localtime(t['timestamp']).tm_yday == time.localtime().tm_yday)
        if today_trades >= self.max_daily_trades:
            raise PermissionError(f"Limite de trades journaliers atteinte: {today_trades}")
        # Règle: taille de position
        if abs(action) > self.max_position_size:
            raise ValueError(f"Taille de position trop grande: {action}")
        # Règle: risque (si action est un pourcentage du capital)
        if hasattr(state, 'portfolio_value') and state.portfolio_value > 0:
            risk = abs(action) / state.portfolio_value
            if risk > self.risk_limit:
                raise ValueError(f"Risque excessif: {risk:.2%}")
        return True
    
    def audit_trail(self, limit=100):
        return self.trade_log[-limit:]

# Instance globale partagée
_audit = DecisionAudit()

def governance_audit(func):
    """Décorateur pour auditer et contrôler les décisions de l'IA."""
    @wraps(func)
    def wrapper(self, state, *args, **kwargs):
        action = func(self, state, *args, **kwargs)
        # Vérifier les règles de gouvernance
        _audit.check_governance(state, action)
        # Enregistrer
        _audit.log_decision(state, action)
        return action
    return wrapper

# Exemple d'utilisation (décorer votre méthode principale) :
# class RLStrategy:
#     @governance_audit
#     def get_action(self, state):
#         # ... votre logique ...

# AUTO-IMPL: argus
# Ajouter après les imports existants
import time
import threading
from collections import deque
from typing import Dict, List

class RuntimeMonitor:
    def __init__(self, window_size: int = 60):
        self.latency_buffer = deque(maxlen=window_size)
        self.error_count = 0
        self._lock = threading.Lock()
        self._alert_threshold = 0.5  # secondes
        self._max_errors = 10
        
    def record_execution(self, start_time: float) -> None:
        latency = time.time() - start_time
        with self._lock:
            self.latency_buffer.append(latency)
            if latency > self._alert_threshold:
                self._trigger_alert(f"High latency detected: {latency:.3f}s")
                
    def record_error(self, error_type: str) -> None:
        with self._lock:
            self.error_count += 1
            if self.error_count > self._max_errors:
                self._trigger_alert(f"Critical: {self.error_count} errors in window")
                
    def _trigger_alert(self, message: str) -> None:
        # Implémentez votre système d'alerte (log, email, etc.)
        import logging
        logging.warning(f"[RUNTIME_MONITOR] {message}")
        
    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "avg_latency": sum(self.latency_buffer) / max(len(self.latency_buffer), 1),
                "max_latency": max(self.latency_buffer) if self.latency_buffer else 0,
                "error_count": self.error_count,
                "buffer_size": len(self.latency_buffer)
            }

# Initialisation globale du moniteur
monitor = RuntimeMonitor(window_size=120)

# Exemple d'utilisation dans votre logique existante (à adapter) :
# Dans la méthode execute_trade() ou run_strategy():
#   start = time.time()
#   try:
#       executer votre code de trading
#       monitor.record_execution(start)
#   except Exception as e:
#       monitor.record_error(type(e).__name__)

# Pour vider les stats périodiquement (ajouter dans la boucle principale)
# stats = monitor.get_stats()
# if stats["avg_latency"] > 1.0:
#     print(f"Anomalie détectée: {stats}")

# AUTO-IMPL: ai-governance-finance
# AI Governance Framework for Trading Decisions
# Add this code to strategy_rl.py without modifying existing logic

class AIGovernance:
    """
    Governance framework to enforce rules on RL trading decisions.
    Overridable rules: risk limits, position size, drawdown, regulatory checks.
    """
    def __init__(self, config=None):
        defaults = {
            'max_position_size': 0.1,          # fraction of portfolio
            'max_drawdown': 0.2,               # max allowed drawdown fraction
            'min_confidence': 0.6,             # probability threshold for action
            'allowed_actions': [0, 1, 2],      # e.g., hold, buy, sell
            'regulatory_blacklist': [],        # symbols not allowed
            'cooldown_steps': 5,               # min steps between same direction trades
        }
        self.config = {**defaults, **(config or {})}
        self.last_trade_step = -self.config['cooldown_steps']
        self.last_trade_action = None

    def approve(self, action, state, step, portfolio_value, current_drawdown):
        """
        Validate a proposed action given current state.
        Returns (approved_action, reason) where approved_action is action or a safe fallback.
        """
        # Action space check
        if action not in self.config['allowed_actions']:
            return (0, "Action not allowed")

        # Confidence check (if state includes action probabilities)
        if 'prob' in state and state['prob'][action] < self.config['min_confidence']:
            return (0, "Confidence too low")

        # Position size check (if state includes proposed size)
        if 'size' in state and state['size'] > self.config['max_position_size'] * portfolio_value:
            return (0, "Position size exceeds limit")

        # Drawdown check
        if current_drawdown > self.config['max_drawdown']:
            return (0, "Max drawdown exceeded")

        # Cooldown check (prevent frequent same-direction trades)
        if action != 0 and action == self.last_trade_action and \
           step - self.last_trade_step < self.config['cooldown_steps']:
            return (0, "Cooldown active")

        # Regulatory blacklist
        symbol = state.get('symbol', None)
        if symbol and symbol in self.config['regulatory_blacklist']:
            return (0, "Symbol blacklisted")

        # Update last trade info if action is not hold
        if action != 0:
            self.last_trade_step = step
            self.last_trade_action = action

        return (action, "Approved")

# Integration example: assume RL agent has method choose_action(state)
# In your existing code, replace direct action execution with:
# governance = AIGovernance()  # instantiate once in the class __init__
# action = agent.choose_action(state)
# approved_action, reason = governance.approve(action, state, step, portfolio_value, drawdown)
# Then use approved_action for trading

# AUTO-IMPL: argus
import time
import numpy as np
from collections import deque

class PerformanceMonitor:
    """Real-time monitoring of performance and anomalies."""
    def __init__(self, window_size=100, anomaly_threshold=3.0):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.returns = deque(maxlen=window_size)
        self.equity_curve = []
        self.trade_log = []
        self.start_time = time.time()
        self.anomalies = []
        self.metrics = {}

    def update(self, reward, equity, trade_info=None):
        """Call after each step with reward, equity, optional trade info."""
        self.returns.append(reward)
        self.equity_curve.append(equity)
        if trade_info:
            self.trade_log.append(trade_info)
        self._check_anomalies(reward, equity)
        self._compute_metrics()

    def _check_anomalies(self, reward, equity):
        if len(self.returns) < 10:
            return
        mean = np.mean(self.returns)
        std = np.std(self.returns)
        if std > 0 and abs(reward - mean) > self.anomaly_threshold * std:
            anomaly = {
                'time': time.time() - self.start_time,
                'reward': reward,
                'equity': equity,
                'z_score': (reward - mean) / std
            }
            self.anomalies.append(anomaly)
            print(f"⚠️ Anomaly detected: {anomaly}")

    def _compute_metrics(self):
        if len(self.equity_curve) < 2:
            return
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        self.metrics['total_return'] = (equity[-1] - equity[0]) / equity[0]
        self.metrics['sharpe_ratio'] = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
        self.metrics['max_drawdown'] = np.max(np.maximum.accumulate(equity) - equity) / np.maximum.accumulate(equity)[-1]
        self.metrics['num_trades'] = len(self.trade_log)
        self.metrics['num_anomalies'] = len(self.anomalies)
        self.metrics['uptime'] = time.time() - self.start_time

    def get_report(self):
        return self.metrics

    def reset(self):
        self.returns.clear()
        self.equity_curve.clear()
        self.trade_log.clear()
        self.anomalies.clear()
        self.start_time = time.time()
        self.metrics = {}

# AUTO-IMPL: argus
import time
import numpy as np
from collections import deque

class ArgusOnBee:
    def __init__(self, window_size=100, threshold_std=3.0):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.latency_buffer = deque(maxlen=window_size)
        self.execution_times = {}
        self.symbols = ["USDJPY", "US30", "US100", "GER40"]
        
    def monitor_execution(self, symbol, order_id, start_time):
        if symbol not in self.symbols:
            return
        latency = time.time() - start_time
        self.latency_buffer.append(latency)
        self.execution_times[order_id] = latency
        
        # Anomaly detection
        if len(self.latency_buffer) >= 30:
            mean = np.mean(self.latency_buffer)
            std = np.std(self.latency_buffer)
            if abs(latency - mean) > self.threshold_std * std:
                print(f"⚠ ANOMALY: {symbol} order {order_id} latency {latency:.4f}s (mean {mean:.4f}s, std {std:.4f}s)")
                self.trigger_alert(symbol, order_id, latency)
        return latency

    def trigger_alert(self, symbol, order_id, latency):
        # Placeholder for alert system (email, slack, etc.)
        print(f"🚨 ARGUS ALERT: {symbol} order {order_id} latency {latency:.4f}s exceeds threshold")

    def get_latency_stats(self):
        if len(self.latency_buffer) == 0:
            return {"mean": 0, "std": 0, "max": 0, "min": 0, "count": 0}
        return {
            "mean": np.mean(self.latency_buffer),
            "std": np.std(self.latency_buffer),
            "max": np.max(self.latency_buffer),
            "min": np.min(self.latency_buffer),
            "count": len(self.latency_buffer)
        }

# Initialize global Argus instance
argus = ArgusOnBee(window_size=100, threshold_std=3.0)

# AUTO-IMPL: argus
import json
import logging
from datetime import datetime
from typing import Optional

class ArgusMonitor:
    """Argus integration for live USDJPY/US100 trade monitoring on TheHive"""
    
    def __init__(self, thehive_url: str, api_key: str, alert_template_path: str = "argus_template.json"):
        self.thehive_url = thehive_url.rstrip('/')
        self.api_key = api_key
        self.alert_template = self._load_template(alert_template_path)
        self.logger = logging.getLogger(__name__)
        
    def _load_template(self, path: str) -> dict:
        """Load Argus alert template"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "title": "Argus Trade Alert",
                "description": "Live trade monitoring alert",
                "severity": 2,
                "tags": ["argus", "trading", "usdjpy", "us100"],
                "type": "trading_alert",
                "source": "strategy_rl"
            }
    
    def send_trade_alert(self, trade_data: dict, instruments: list = ["USDJPY", "US100"]) -> Optional[str]:
        """Send trade alert to TheHive via Argus"""
        try:
            alert = self.alert_template.copy()
            alert.update({
                "title": f"Argus Alert: {trade_data.get('action', 'UNKNOWN')} {', '.join(instruments)}",
                "description": json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "instruments": instruments,
                    "trade": trade_data,
                    "status": "live_monitoring"
                }),
                "date": int(datetime.utcnow().timestamp() * 1000)
            })
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            import requests
            response = requests.post(
                f"{self.thehive_url}/api/v1/alert",
                json=alert,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            alert_id = response.json().get("id")
            self.logger.info(f"Argus alert sent to TheHive: {alert_id}")
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Failed to send Argus alert: {e}")
            return None

# Initialization example (add to your existing strategy_rl.py initialization)
def init_argus_monitor(config: dict) -> ArgusMonitor:
    """Initialize Argus monitor with TheHive config"""
    return ArgusMonitor(
        thehive_url=config.get("thehive_url", "http://localhost:9000"),
        api_key=config.get("argus_api_key", ""),
        alert_template_path=config.get("argus_template", "argus_template.json")
    )

# AUTO-IMPL: rl-reward-structure
# Ajouter cette fonction dans strategy_rl.py
def optimize_reward_multi_asset(rewards, weights, risk_free_rate=0.02, lambda_sharpe=0.5):
    """
    Optimise la récompense pour un agent RL multi-actifs.
    Combine Sharpe ratio ajusté, pénalité de turnover et diversification.
    """
    import numpy as np
    
    rewards = np.array(rewards)
    weights = np.array(weights)
    
    # 1. Sharpe ratio ajusté par actif
    excess_returns = rewards - risk_free_rate / 252  # quotidien
    sharpe_per_asset = np.mean(excess_returns, axis=0) / (np.std(excess_returns, axis=0) + 1e-8)
    
    # 2. Pénalité de turnover (changement de poids)
    turnover_penalty = -0.01 * np.sum(np.abs(np.diff(weights, axis=0)), axis=1).mean()
    
    # 3. Bonus de diversification (entropie des poids)
    if len(weights.shape) > 1:
        weights_norm = np.clip(weights[-1], 1e-8, 1 - 1e-8)  # éviter log(0)
        weights_norm = weights_norm / np.sum(weights_norm)
        diversification_bonus = np.exp(np.sum(-weights_norm * np.log(weights_norm)))
    else:
        diversification_bonus = 0.0
    
    # 4. Récompense composite
    reward = (lambda_sharpe * sharpe_per_asset.mean() +
              (1 - lambda_sharpe) * rewards.mean() +
              turnover_penalty + 
              0.05 * diversification_bonus)
    
    return float(reward)

# Modifier la méthode compute_reward dans la classe RLAgent
# (suppose que la classe existante a cette méthode)
def compute_reward_with_optimization(self, actions, prices, weights):
    """
    Remplacez l'appel à compute_reward d'origine dans votre boucle d'entraînement.
    """
    rewards = self.compute_reward(actions, prices)  # méthode existante
    optimized_reward = optimize_reward_multi_asset(
        rewards, 
        weights,
        risk_free_rate=0.02,
        lambda_sharpe=0.5
    )
    return optimized_reward

# AUTO-IMPL: rl-reward-structure
import numpy as np

def _calculate_multi_asset_reward(self, actions, portfolio_values, prev_portfolio_values):
    """
    Optimized reward function for multi-asset RL agent.
    Combines profit, volatility penalty, and diversification bonus.
    """
    # Profit component (weighted return)
    returns = (portfolio_values - prev_portfolio_values) / (prev_portfolio_values + 1e-8)
    profit_reward = np.sum(returns * self.asset_weights)  # asset_weights: predefined or adaptive

    # Volatility penalty (reduce risk taking)
    portfolio_std = np.std(portfolio_values[-self.lookback:]) if len(portfolio_values) >= self.lookback else 0
    volatility_penalty = -self.volatility_coef * portfolio_std  # self.volatility_coef > 0

    # Diversification bonus (encourage spread across assets)
    if len(actions) > 1:
        asset_allocation = np.abs(actions) / (np.sum(np.abs(actions)) + 1e-8)
        diversification = 1 - np.sum(asset_allocation ** 2)  # Herfindahl index inverse
    else:
        diversification = 0
    diversification_bonus = self.diversification_coef * diversification

    # Risk-adjusted return (Sharpe-like but incremental)
    risk_free_rate = 0.02 / 252  # daily approx
    excess_return = profit_reward - risk_free_rate
    if portfolio_std > 0:
        sharpe_component = self.sharpe_coef * (excess_return / (portfolio_std + 1e-8))
    else:
        sharpe_component = 0

    # Total reward
    total_reward = profit_reward + volatility_penalty + diversification_bonus + sharpe_component
    return total_reward

# Insert into training loop where reward is computed
# e.g., inside step function: reward = self._calculate_multi_asset_reward(...)

# AUTO-IMPL: rl-reward-structure
# Nouvelle fonction de récompense optimisée pour le ratio profit/risque (test DeepSeek V4)
# Ajoutez-la à la fin de strategy_rl.py

import numpy as np  # déjà présent normalement

def reward_profit_risk(returns_buffer, window=10, risk_free=0.0, scale_factor=1.0):
    """
    Calcule une récompense basée sur le ratio de Sharpe d'une fenêtre glissante.
    Améliore la stabilité en pénalisant la volatilité excessive.

    Paramètres:
        returns_buffer (list ou array): Historique des rendements (ex: daily returns).
        window (int): Taille de la fenêtre pour le calcul du risque.
        risk_free (float): Taux sans risque annualisé ajusté à la période.
        scale_factor (float): Facteur de mise à l'échelle de la récompense.

    Retourne:
        float: Récompense scalée entre -1 et 1 (via tanh).
    """
    if len(returns_buffer) < 2:
        return 0.0

    # Prendre les rendements les plus récents dans la fenêtre
    recent = returns_buffer[-window:] if len(returns_buffer) >= window else returns_buffer

    mean_ret = np.mean(recent)
    std_ret = np.std(recent, ddof=1)

    if std_ret == 0:
        return 0.0

    sharpe = (mean_ret - risk_free) / std_ret

    # Appliquer tanh pour bornier la récompense et stabiliser l'apprentissage
    reward = np.tanh(sharpe * scale_factor)
    return reward

# Exemple d'intégration (à décommenter et adapter dans votre environnement):
# Dans la méthode step() de l'environnement, remplacez l'ancienne récompense par:
#   self.returns_history.append(reward)  # où reward = profit de l'étape
#   rl_reward = reward_profit_risk(self.returns_history, window=10)

# AUTO-IMPL: argus
import time
import statistics
from collections import deque
from functools import wraps

class RuntimeMonitor:
    """Adds runtime monitoring for latency, errors, and anomalies."""
    
    def __init__(self, window_size=100, anomaly_std_factor=3.0):
        self.window_size = window_size
        self.anomaly_std_factor = anomaly_std_factor
        self.latencies = deque(maxlen=window_size)
        self.error_count = 0
        self.total_calls = 0
        
    def record_latency(self, latency):
        self.latencies.append(latency)
        self._detect_anomaly(latency)
        
    def _detect_anomaly(self, value):
        if len(self.latencies) < 2:
            return
        mean = statistics.mean(self.latencies)
        std = statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0.0
        if std > 0 and abs(value - mean) > self.anomaly_std_factor * std:
            print(f"Anomaly detected: latency {value:.3f}s (mean={mean:.3f}s, std={std:.3f}s)")
            
    def record_error(self):
        self.error_count += 1
        
    def get_stats(self):
        return {
            'avg_latency': statistics.mean(self.latencies) if self.latencies else 0.0,
            'max_latency': max(self.latencies) if self.latencies else 0.0,
            'min_latency': min(self.latencies) if self.latencies else 0.0,
            'error_rate': self.error_count / max(self.total_calls, 1),
            'total_calls': self.total_calls
        }

def monitor_operation(monitor: RuntimeMonitor):
    """Decorator to wrap functions with latency & error monitoring."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                monitor.record_error()
                raise e
            finally:
                elapsed = time.perf_counter() - start
                monitor.total_calls += 1
                monitor.record_latency(elapsed)
        return wrapper
    return decorator

# Example usage (uncomment if needed):
# monitor = RuntimeMonitor(window_size=50)
# @monitor_operation(monitor)
# def my_action(state):
#     # ... trading logic
#     pass

# AUTO-IMPL: rl-reward-structure
# Ajouter à la fin de strategy_rl.py, avant l'usage de la récompense

def optimize_reward_function(reward_dict, config):
    """
    Optimise la fonction de récompense pour aligner sur les objectifs de trading.
    Ajoute des composantes ajustables : profit, risque, drawdown, slippage.
    """
    profit = reward_dict.get('profit', 0.0)
    risk = reward_dict.get('risk', 0.0)
    max_drawdown = reward_dict.get('max_drawdown', 0.0)
    slippage_cost = reward_dict.get('slippage_cost', 0.0)

    # Poids configurables
    w_profit = config.get('w_profit', 1.0)
    w_risk = config.get('w_risk', -0.5)
    w_drawdown = config.get('w_drawdown', -0.3)
    w_slippage = config.get('w_slippage', -0.2)

    # Pénalité exponentielle pour drawdown sévère
    drawdown_penalty = -w_drawdown * (max_drawdown ** 2) if max_drawdown > 0.05 else 0.0

    # Récompense ajustée
    reward = (w_profit * profit
              + w_risk * risk
              + drawdown_penalty
              + w_slippage * slippage_cost)

    # Normalisation douce (optionnelle)
    reward = max(min(reward, 10.0), -10.0)

    return reward

# Exemple d'intégration dans la boucle d'apprentissage (à adapter selon votre code)
# Dans la fonction compute_reward() existante, remplacer le return par:
# if hasattr(self, 'config_rl'):
#     return optimize_reward_function(locals(), self.config_rl)
# else:
#     return original_reward

# AUTO-IMPL: rl-reward-structure
import numpy as np
from typing import Optional

def risk_adjusted_reward(returns: np.ndarray,
                         costs: np.ndarray,
                         risk_free_rate: float = 0.0,
                         gamma: float = 1.0,
                         lambda_cost: float = 1.0) -> float:
    """
    Compute a reward that balances risk-adjusted returns and transaction costs.

    Reward = (mean(returns) - risk_free_rate) / (std(returns) + 1e-8) * gamma
             - lambda_cost * np.sum(np.abs(costs))

    Args:
        returns: Array of recent portfolio returns.
        costs: Array of transaction costs (e.g., spread + slippage) per step.
        risk_free_rate: Annual risk-free rate, converted to step rate.
        gamma: Scaling factor for risk-adjusted term.
        lambda_cost: Penalty weight for total transaction costs.

    Returns:
        Scalar reward.
    """
    if len(returns) == 0:
        return 0.0
    mean_ret = np.mean(returns) - risk_free_rate
    std_ret  = np.std(returns) + 1e-8  # avoid division by zero
    sharpe   = mean_ret / std_ret
    cost_penalty = lambda_cost * np.sum(np.abs(costs))
    return gamma * sharpe - cost_penalty


class RiskAwareRewardMixin:
    """Mixin to replace base reward calculation with risk-adjusted metric."""

    def calculate_reward(self, returns: np.ndarray,
                         costs: np.ndarray,
                         **kwargs) -> float:
        """Override this method in your RL agent to use the new reward."""
        return risk_adjusted_reward(returns, costs, **kwargs)

# AUTO-IMPL: ai-governance-finance
# À ajouter dans strategy_rl.py, par exemple dans la classe RLStrategy ou en tant que fonctions séparées.

class RegulatoryValidator:
    """
    Couche de validation et de contraintes réglementaires.
    Peut être intégrée avant l'exécution des actions du RL.
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.max_position = self.config.get('max_position', 1000)
        self.max_order_size = self.config.get('max_order_size', 200)
        self.trading_hours = self.config.get('trading_hours', (9, 17))
        self.short_selling_allowed = self.config.get('short_selling_allowed', True)
        self.leverage_limit = self.config.get('leverage_limit', 2.0)

    def validate_action(self, action: dict, state: dict) -> bool:
        """
        Valide une action avant exécution.
        Retourne True si l'action est autorisée, False sinon.
        """
        # Vérification des heures de trading
        current_hour = state.get('current_hour', 0)
        if not (self.trading_hours[0] <= current_hour < self.trading_hours[1]):
            print(f"Action refusée : hors heures de trading ({self.trading_hours})")
            return False

        # Vérification de la taille de l'ordre
        order_size = action.get('size', 0)
        if order_size > self.max_order_size:
            print(f"Action refusée : taille d'ordre {order_size} > max {self.max_order_size}")
            return False

        # Vérification de la limite de position
        current_position = state.get('position', 0)
        new_position = current_position + action.get('direction', 0) * order_size
        if abs(new_position) > self.max_position:
            print(f"Action refusée : position maximale dépassée ({abs(new_position)} > {self.max_position})")
            return False

        # Vérification de l'autorisation de vente à découvert
        if action.get('direction', 0) < 0 and not self.short_selling_allowed:
            print("Action refusée : vente à découvert interdite")
            return False

        # Vérification du levier
        capital = state.get('capital', 1.0)
        if capital > 0 and abs(new_position) / capital > self.leverage_limit:
            print(f"Action refusée : levier {abs(new_position)/capital:.2f} > max {self.leverage_limit}")
            return False

        return True

# Exemple d'utilisation (à intégrer dans la boucle d'apprentissage)
# validator = RegulatoryValidator(config)
# if validator.validate_action(action, state):
#     exécuter_action(action)
# else:
#     action = action_par_défaut_ou_zero


# AUTO-IMPL: worldcycle
# Integration with WorldCycle + DeepSeek V4 via OpenRouter
# FIXED: asyncio.run() -> loop reuse, eval() -> json/ast.literal_eval,
#        deprecated OpenAI API, missing _execute_trade, GPU configurable


import os
import ast
import json
import asyncio
import torch
from typing import Dict, Any
import numpy as np
from openai import AsyncOpenAI
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

_openai_client = None
def _get_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    return _openai_client

DEEPSEEK_MODEL = "deepseek/deepseek-v4"
DEFAULT_GPU = int(os.environ.get("JEPA_GPU", "1"))

class WorldCycleDeepSeekPolicy:
    """RL policy with WorldCycle state representation and DeepSeek action suggestions"""

    def __init__(self, env_config: Dict[str, Any], n_envs: int = 8):
        self.n_envs = n_envs
        self.env = SubprocVecEnv([lambda: self._make_env(env_config) for _ in range(n_envs)])
        self.model = PPO("MlpPolicy", self.env, verbose=1, learning_rate=3e-4, n_steps=2048)
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    def _make_env(self, config: Dict):
        try:
            from gymnasium import Env
            from gymnasium.spaces import Box
        except ImportError:
            from gym import Env
            from gym.spaces import Box

        class WorldCycleEnv(Env):
            def __init__(self, cfg):
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=(20,))
                self.action_space = Box(low=-1, high=1, shape=(3,))

            def step(self, action):
                state = self._get_world_state()
                obs = self._encode_observation(state)
                deepseek_action = self._query_deepseek_sync(state)
                alpha = float(os.environ.get("DEEPSEEK_BLEND", "0.3"))
                blended = (1 - alpha) * action + alpha * deepseek_action
                blended = np.clip(blended, -1, 1)
                reward = self._execute_trade(blended)
                done = False
                info = {"deepseek_advice": deepseek_action}
                return obs, reward, done, info

            def _get_world_state(self):
                # TODO: replace with real market data + cycle indicators
                return np.random.rand(10)

            def _encode_observation(self, state):
                return np.concatenate([state, [np.mean(state), np.std(state), state[-1]]])

            @staticmethod
            def _query_deepseek_sync(state_array):
                """Sync wrapper - avoids nested asyncio.run() RuntimeError"""
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(
                    WorldCycleEnv._query_deepseek_async(state_array)
                )

            @staticmethod
            async def _query_deepseek_async(state_array):
                """Query DeepSeek V4 via OpenRouter (async) - safe parsing, no eval()"""
                try:
                    client = _get_client()
                    response = await client.chat.completions.create(
                        model=DEEPSEEK_MODEL,
                        messages=[{
                            "role": "system",
                            "content": "You are a trading advisor. Respond with a JSON array of 3 floats "
                                       "in range [-1,1] for position, size, leverage. Example: [0.5,0.3,-0.2]"}
                        ],
                        max_tokens=50
                    )
                    content = response.choices[0].message.content.strip()
                    try:
                        advice = np.array(json.loads(content))
                    except (json.JSONDecodeError, ValueError):
                        try:
                            advice = np.array(ast.literal_eval(content))
                        except (ValueError, SyntaxError):
                            return np.zeros(3)
                    if advice.shape != (3,):
                        return np.zeros(3)
                    return np.clip(advice, -1, 1).astype(np.float32)
                except Exception as e:
                    print(f"DeepSeek query failed: {e}")
                    return np.zeros(3)

            @staticmethod
            def _execute_trade(action):
                """Execute trade - reward proxy. TODO: replace with real MT5 bridge call"""
                return float(np.tanh(action[0]) * 0.01)

        return WorldCycleEnv(config)

    def train(self, total_timesteps: int = 500000):
        gpu_id = DEFAULT_GPU
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        action, _states = self.model.predict(observation, state, episode_start, deterministic)
        alpha = float(os.environ.get("DEEPSEEK_BLEND", "0.3"))
        return (1 - alpha) * action + alpha * np.zeros(3), _states


# Usage example
if __name__ == "__main__":
    config = {"symbol": "BTC/USD", "lookback": 100}
    agent = WorldCycleDeepSeekPolicy(config)
    agent.train()

# AUTO-IMPL: argus
# Add to strategy_rl.py - Argus monitoring integration for live trading loops

import threading
import time
from datetime import datetime
import logging
import json

# Argus monitoring class for live trade oversight
class ArgusMonitor:
    def __init__(self, symbols, deepseek_v4_thresholds=None):
        self.symbols = symbols  # ['USDJPY', 'US30', 'US100', 'GER40']
        self.thresholds = deepseek_v4_thresholds or {
            'min_profit_pips': 5,
            'max_loss_pips': -15,
            'max_lot_size': 0.5,
            'max_open_trades': 5,
            'anomaly_deviation': 2.5
        }
        self.active = False
        self.trade_log = []
        self.logger = logging.getLogger('Argus')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - ARGUS - %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def monitor_loop(self, get_current_positions_func, place_trade_func=None):
        """Continuous monitoring thread for live trading anomalies"""
        self.active = True
        while self.active:
            try:
                for symbol in self.symbols:
                    positions = get_current_positions_func(symbol)
                    if not positions:
                        continue
                    
                    total_profit = sum(p['profit'] for p in positions)
                    open_count = len(positions)
                    
                    # DeepSeek V4 threshold checks
                    if total_profit < self.thresholds['max_loss_pips']:
                        self.logger.warning(f"ANOMALY {symbol}: Loss threshold exceeded - {total_profit} pips")
                        # Optional auto-hedge logic here
                        self._flag_anomaly(symbol, 'excessive_loss', total_profit)
                    
                    if open_count > self.thresholds['max_open_trades']:
                        self.logger.error(f"ANOMALY {symbol}: Over-trading - {open_count} open positions")
                        self._flag_anomaly(symbol, 'over_trading', open_count)
                    
                    # Deviation check (simplified DeepSeek V4)
                    avg_profit = total_profit / max(open_count, 1)
                    if abs(avg_profit) > self.thresholds['anomaly_deviation'] * 2:
                        self.logger.info(f"ARGUS {symbol}: Deviation alert - avg profit {avg_profit:.2f}")
                
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.logger.error(f"Argus monitor crashed: {e}")
                time.sleep(10)

    def _flag_anomaly(self, symbol, anomaly_type, value):
        """Log anomaly with timestamp for later analysis"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'type': anomaly_type,
            'value': value,
            'threshold': self.thresholds.get(f'max_{anomaly_type.replace("_", "_")}')
        }
        self.trade_log.append(entry)
        # Can also trigger Telegram/email alerts here

    def start(self, get_positions_func):
        thread = threading.Thread(target=self.monitor_loop, args=(get_positions_func,))
        thread.daemon = True
        thread.start()
        self.logger.info(f"Argus deployed on Bee for {', '.join(self.symbols)}")
        return thread

    def stop(self):
        self.active = False
        self.logger.info("Argus monitoring stopped")

# Usage example (to be placed inside your RL strategy class init or main loop):
# argus = ArgusMonitor(['USDJPY', 'US30', 'US100', 'GER40'])
# argus.start(self.get_positions)  # where self.get_positions returns list of dicts with 'profit'

# AUTO-IMPL: ai-governance-finance
# guardrails.py (ou ajout à strategy_rl.py)
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("DeepSeekAudit")

class BeeGuardrails:
    """Rule-based guardrails for position limits and drawdown caps."""
    def __init__(self, max_position: float = 1.0, max_drawdown: float = 0.2):
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        self.equity_peak = None

    def check(self, equity: float, position: float) -> bool:
        """Return False if trade should be blocked.
        position in % of equity, drawdown vs peak."""
        if self.equity_peak is None:
            self.equity_peak = equity
        self.equity_peak = max(self.equity_peak, equity)
        drawdown = (self.equity_peak - equity) / self.equity_peak if self.equity_peak else 0
        if drawdown > self.max_drawdown:
            audit_decision("blocked", f"Drawdown {drawdown:.2%} > {self.max_drawdown:.2%}")
            return False
        if abs(position) > self.max_position:
            audit_decision("blocked", f"Position {position:.2f} > {self.max_position:.2f}")
            return False
        return True

def audit_decision(action: str, reason: str = "", context: Dict[str, Any] = None):
    """Log explainability for DeepSeek V4 decisions."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "reason": reason,
        "context": context or {},
        "model": "DeepSeek V4",
    }
    logger.info(f"AUDIT: {log_entry}")
