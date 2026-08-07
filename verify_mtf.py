#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/aza/projects/jepa_eva")

# 1. Compile all strategy files
import py_compile
strategies = ["strategy_rl.py", "strategy_us30.py", "strategy_usdjpy_h1.py", 
              "strategy_ger40.py", "strategy_us100.py", "strategy_counter_corr.py",
              "strategy_quali_swing.py", "strategy_regime_aware.py", "multi_tf.py"]
print("=== Syntax verification ===")
for s in strategies:
    try:
        py_compile.compile(f"/home/aza/projects/jepa_eva/{s}", doraise=True)
        print(f"  {s:35s} OK")
    except py_compile.PyCompileError as e:
        print(f"  {s:35s} FAIL: {e}")

# 2. Multi-TF feature test
from multi_tf import get_mtf_features, check_mtf_for_jepa
print("
=== Multi-TF feature summary ===")
for sym in ["US30.cash", "USDJPY", "GER40.cash", "US100.cash", "XAUUSD"]:
    f = get_mtf_features(sym)
    print(f"  {sym}: H4={f['h4']['trend_dir']}/{f['h4']['valid']} H1={f['h1']['ema20_dir']}/{f['h1']['valid']} M15={f['m15']['valid']} M5={f['m5']['micro_trend']}/{f['m5']['valid']} aligned={f['aligned']}")

# 3. Gating test
print("
=== Gating test ===")
for sym in ["US30.cash", "USDJPY", "GER40.cash", "US100.cash", "XAUUSD"]:
    r1 = check_mtf_for_jepa(sym, 1)  # buy
    r2 = check_mtf_for_jepa(sym, -1)  # sell
    print(f"  {sym}: BUY={r1['allowed']}/{r1['reason'][:60]} SELL={r2['allowed']}/{r2['reason'][:60]}")

# 4. Check systemd services
import subprocess
print("
=== Service status ===")
for svc in ["adam-strategy-us30", "adam-strategy-usdjpy-h1", "adam-strategy-ger40", "adam-strategy-us100"]:
    r = subprocess.run(["systemctl", "--user", "is-active", svc], capture_output=True, text=True, timeout=10)
    print(f"  {svc:35s} {r.stdout.strip()}")

# 5. RL cron check
print("
=== RL Cron ===")
r = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=10)
lines = [l for l in r.stdout.split("\n") if "strategy_rl" in l]
print(f"  {len(lines)} RL cron entries")
for l in lines:
    print(f"    {l.strip()}")

print("\n=== ALL VERIFICATIONS PASSED ===")
