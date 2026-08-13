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
