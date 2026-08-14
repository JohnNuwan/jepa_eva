#!/usr/bin/env python3
"""
EVA LEARN — Boucle d'apprentissage continu
Collecte les trades fermés, ré-entraîne, swap si meilleur
"""
import sys, os, json, time, sqlite3, urllib.request, subprocess
from datetime import datetime, timedelta
from pathlib import Path

JEPA_DIR = "/home/aza/projects/jepa_eva"
VENV = "/home/aza/jepa_eva/venv/bin/python3"
LOG_DIR = "/home/aza/eva-adam-v2/logs"
DB_PATH = f"{JEPA_DIR}/eva_learn.db"
BRIDGE_URL = "http://192.168.1.6:8765"

os.chdir(JEPA_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def log(msg):
    print(f"[{datetime.now().isoformat()}] {msg}")
    sys.stdout.flush()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS trades (
        ticket INTEGER PRIMARY KEY,
        symbol TEXT, type TEXT, volume REAL, open_price REAL, close_price REAL,
        profit REAL, swap REAL, commission REAL, sl REAL, tp REAL,
        open_time TEXT, close_time TEXT, comment TEXT, source TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS positions_live (
        ticket INTEGER PRIMARY KEY,
        symbol TEXT, type TEXT, volume REAL, open_price REAL,
        profit REAL, open_time TEXT, comment TEXT, updated_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS champions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT, deployed_at TEXT, profit_factor REAL, drawdown REAL,
        holdout_return REAL, total_trades INTEGER, win_rate REAL,
        champion_file TEXT, active INTEGER DEFAULT 1
    )""")
    conn.commit()
    return conn

def fetch_closed_trades():
    """Récupère les trades fermés depuis MT5 Bridge"""
    try:
        req = urllib.request.Request(f"{BRIDGE_URL}/history", method="GET")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        log(f"⚠ Erreur fetch history: {e}")
        return None

def fetch_positions():
    """Récupère les positions ouvertes"""
    try:
        req = urllib.request.Request(f"{BRIDGE_URL}/positions", method="GET")
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        log(f"⚠ Erreur fetch positions: {e}")
        return None

def store_trades(conn, trades):
    """Stocke les trades fermés dans la DB"""
    if not trades: return 0
    count = 0
    for t in trades:
        try:
            cur = conn.execute("""INSERT OR IGNORE INTO trades 
                (ticket, symbol, type, volume, profit, swap, commission, close_time, comment, source)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (int(t.get("ticket", 0)), t.get("symbol", ""), t.get("type", ""), float(t.get("volume", 0)),
                 float(t.get("profit", 0)), float(t.get("swap", 0)), float(t.get("commission", 0)),
                 t.get("time", ""), t.get("comment"), "mt5"))
            if cur.rowcount > 0:
                count += 1
        except Exception as e:
            log(f"  ⚠ Insert error: {e}")
    conn.commit()
    return count


def store_positions(conn, positions):
    """Stocke les positions ouvertes (P&L flottant)"""
    if not positions:
        return 0
    conn.execute("DELETE FROM positions_live")
    count = 0
    for p in positions:
        try:
            conn.execute("""INSERT OR REPLACE INTO positions_live
                (ticket, symbol, type, volume, open_price, profit, open_time, comment, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (int(p.get("ticket", 0)), p.get("symbol", ""), p.get("type", ""),
                 float(p.get("volume", 0)), float(p.get("open_price", 0)),
                 float(p.get("profit", 0)), str(p.get("open_time", "")),
                 p.get("comment", ""), datetime.now().isoformat()))
            count += 1
        except Exception as e:
            log(f"  ⚠ Position insert error: {e}")
    conn.commit()
    return count

def get_float_pnl(conn):
    """Calcule le P&L flottant total des positions ouvertes"""
    cur = conn.execute("SELECT COALESCE(SUM(profit),0), COUNT(*) FROM positions_live")
    row = cur.fetchone()
    return (round(row[0], 2), row[1]) if row else (0.0, 0)

def evaluate_current_champion():
    """Calcule les métriques du champion actuel depuis les trades réels"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("""SELECT COUNT(*), COALESCE(SUM(profit),0), COALESCE(AVG(profit),0),
        SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
        MIN(profit) as max_loss
        FROM trades WHERE source='mt5' AND symbol='XAUUSD'""")
    row = cur.fetchone()
    conn.close()
    if not row or row[0] == 0:
        return None
    total, total_pl, avg_pl, wins, max_loss = row
    losses = total - wins
    pf = abs(total_pl / max_loss) if max_loss < 0 else total_pl if total_pl > 0 else 0
    wr = wins / total if total > 0 else 0
    return {
        "total_trades": total, "total_pl": round(total_pl, 2),
        "avg_pl": round(avg_pl, 2), "profit_factor": round(pf, 2),
        "win_rate": round(wr, 3), "max_loss": round(max_loss, 2)
    }

def get_current_champion_file():
    """Trouve le champion actuellement déployé"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT champion_file FROM champions WHERE active=1 ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def deploy_champion(champion_file, metrics):
    """Déploie un nouveau champion (swap model + restart JEPA)"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE champions SET active=0 WHERE active=1")
    conn.execute("""INSERT INTO champions (symbol, deployed_at, profit_factor, drawdown,
        holdout_return, total_trades, win_rate, champion_file, active)
        VALUES (?,?,?,?,?,?,?,?,1)""",
        ("XAUUSD", datetime.now().isoformat(), metrics.get("profit_factor", 0),
         metrics.get("drawdown", 0), metrics.get("holdout_return", 0),
         metrics.get("total_trades", 0), metrics.get("win_rate", 0), champion_file))
    conn.commit()
    conn.close()
    # Restart JEPA to load new model
    subprocess.run(["systemctl", "--user", "restart", "adam-jepa.service"], 
                   capture_output=True, timeout=30)
    log(f"🚀 Nouveau champion déployé: {champion_file}")

def main():
    log("=" * 60)
    log("EVA LEARN — Boucle d'apprentissage continu")
    log("=" * 60)
    
    conn = init_db()
    os.makedirs(LOG_DIR, exist_ok=True)
    
    cycle = 0
    while True:
        cycle += 1
        log(f"\n--- Cycle #{cycle} ---")
        
        # 1. Collect trades from MT5
        trades = fetch_closed_trades()
        if trades and isinstance(trades, dict):
            trades = trades.get('deals', [])
        if trades:
            count = store_trades(conn, trades)
            if count > 0:
                log(f"  ✅ {count} nouveaux trades stockés")
        
        
        # 1b. Suivi positions ouvertes (P&L flottant)
        positions = fetch_positions()
        if positions and isinstance(positions, dict):
            pos_list = positions.get('positions', [])
            store_positions(conn, pos_list)
            fpnl, npos = get_float_pnl(conn)
            if npos > 0:
                log(f"  # Positions live: {npos} | P&L flottant: {fpnl:+.2f}$")
            else:
                log(f"  # Aucune position ouverte")

        # 2. Evaluate current performance
        metrics = evaluate_current_champion()
        if metrics:
            log(f"  📊 Performance actuelle:")
            log(f"     Trades: {metrics['total_trades']} | P/L: {metrics['total_pl']}$")
            log(f"     Profit Factor: {metrics['profit_factor']} | Win Rate: {metrics['win_rate']:.1%}")
            log(f"     Avg P/L: {metrics['avg_pl']}$ | Max Loss: {metrics['max_loss']}$")
        
        # 3. Every 10 cycles, check if we have enough data to retrain
        if cycle % 5 == 0 and metrics and metrics['total_trades'] >= 5:
            log(f"  🔄 Lancement validation avec {metrics['total_trades']} trades réels...")
            rc = subprocess.run(
                [VENV, "train_arena_validated.py", "--symbole", "XAUUSD", 
                 "--generations", "50"],
                capture_output=True, text=True, timeout=7200
            )
            if rc.returncode == 0:
                log(f"  ✅ Validation terminée")
                # Check for new champions
                import glob
                registries = glob.glob("registry_arena_validated/XAUUSD_registry.jsonl")
                if registries:
                    with open(registries[0]) as f:
                        candidates = [json.loads(l) for l in f.readlines() if l.strip()]
                    for c in candidates:
                        pf = c.get("profit_factor", 0)
                        if pf > metrics["profit_factor"]:
                            log(f"  🏆 Nouveau champion détecté! PF={pf} > {metrics['profit_factor']}")
                            deploy_champion(registries[0], c)
                            break
            else:
                log(f"  ❌ Validation échouée: {rc.stderr[-200:]}")
        
        # 4. Sleep 1 hour between cycles
        log(f"  💤 Prochain cycle dans 1h...")
        time.sleep(300)

if __name__ == "__main__":
    main()