#!/usr/bin/env python3
"""Extract top 20 champions from registry_massive (TheHive + Bee)."""
import json, os, sys, subprocess
from pathlib import Path

THEHIVE = "/home/aza/projects/jepa_eva"
BEE_BASE = "/home/debia/jepa_eva"

def parse_registry_jsonl(path):
    entries = []
    if not os.path.isfile(path):
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except:
                    pass
    return entries

def extract_hive_champions():
    all_entries = []
    base = os.path.join(THEHIVE, "registry_massive")
    if not os.path.isdir(base):
        print(f"WARNING: {base} not found")
        return all_entries
    for run_dir in sorted(os.listdir(base)):
        run_path = os.path.join(base, run_dir)
        if not os.path.isdir(run_path):
            continue
        registry_file = os.path.join(run_path, "registry.jsonl")
        champions_dir = os.path.join(run_path, "champions")
        if not os.path.isfile(registry_file):
            continue
        entries = parse_registry_jsonl(registry_file)
        for e in entries:
            parts = run_dir.split("_")
            symbol = parts[0]
            if len(parts) >= 2 and parts[1] == "cash":
                symbol = f"{parts[0]}.cash"
            elif len(parts) >= 2 and parts[1] in ["baseline","biggru","calmar","exploit","explore","highcost","longseg","lowcost","pf","seed7","sharpe","shortseg","smallgru","wf3","wr"]:
                symbol = parts[0]
            champ_file = e.get("fichier", "")
            champ_path = os.path.join(champions_dir, champ_file) if champ_file else ""
            all_entries.append({
                "symbol": symbol,
                "run_id": run_dir,
                "variant": "_".join(parts[1:]) if len(parts) > 1 else "default",
                "champion_path": champ_path if os.path.isfile(champ_path) else "",
                "fitness": e.get("fitness", -999),
                "net_profit": e.get("net_profit", 0),
                "drawdown": e.get("drawdown", 100),
                "sortino": e.get("sortino", 0),
                "win_rate": e.get("win_rate", 0),
                "profit_factor": e.get("profit_factor", 0),
                "nb_trades": e.get("nb_trades", 0),
                "fitness_train": e.get("fitness_train", 0),
                "generation": e.get("generation", 0),
                "fold": e.get("fold", "unknown"),
                "source": "hive",
            })
    return all_entries

def extract_bee_champions():
    all_entries = []
    try:
        cmd = ["docker","exec","wg-easy","sshpass","-p","thWrzIUh",
               "ssh","-o","StrictHostKeyChecking=no","-o","ConnectTimeout=10",
               "debia@10.8.0.4",
               f"find {BEE_BASE}/registry_massive -name 'registry.jsonl' 2>/dev/null"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0 or not r.stdout.strip():
            print(f"Bee registry files: ERROR or empty: {r.stderr[:200]}")
            return all_entries
        reg_files = [f.strip() for f in r.stdout.strip().split("\n") if f.strip()]
        print(f"Found {len(reg_files)} registry files on Bee")
        for reg_file in reg_files:
            read_cmd = ["docker","exec","wg-easy","sshpass","-p","thWrzIUh",
                        "ssh","-o","StrictHostKeyChecking=no","-o","ConnectTimeout=10",
                        "debia@10.8.0.4", f"cat {reg_file}"]
            r2 = subprocess.run(read_cmd, capture_output=True, text=True, timeout=30)
            if r2.returncode != 0 or not r2.stdout.strip():
                continue
            run_dir = os.path.basename(os.path.dirname(reg_file))
            champions_dir = os.path.join(os.path.dirname(reg_file), "champions")
            champ_cmd = ["docker","exec","wg-easy","sshpass","-p","thWrzIUh",
                         "ssh","-o","StrictHostKeyChecking=no","-o","ConnectTimeout=10",
                         "debia@10.8.0.4", f"ls {champions_dir}/ 2>/dev/null"]
            r3 = subprocess.run(champ_cmd, capture_output=True, text=True, timeout=15)
            champ_files = [c.strip() for c in r3.stdout.strip().split("\n") if c.strip()] if r3.stdout.strip() else []
            for line in r2.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except:
                    continue
                parts = run_dir.split("_")
                symbol = parts[0]
                if len(parts) >= 2 and parts[1] == "cash":
                    symbol = f"{parts[0]}.cash"
                elif len(parts) >= 2 and parts[1] in ["baseline","biggru","calmar","exploit","explore","highcost","longseg","lowcost","pf","seed7","sharpe","shortseg","smallgru","wf3","wr","r1","cpu"]:
                    symbol = parts[0]
                champ_file = e.get("fichier", "")
                champ_exists = champ_file in champ_files if champ_file else False
                all_entries.append({
                    "symbol": symbol,
                    "run_id": run_dir,
                    "variant": "_".join(parts[1:]) if len(parts) > 1 else "default",
                    "champion_path": f"{champions_dir}/{champ_file}" if champ_file and champ_exists else "",
                    "fitness": e.get("fitness", -999),
                    "net_profit": e.get("net_profit", 0),
                    "drawdown": e.get("drawdown", 100),
                    "sortino": e.get("sortino", 0),
                    "win_rate": e.get("win_rate", 0),
                    "profit_factor": e.get("profit_factor", 0),
                    "nb_trades": e.get("nb_trades", 0),
                    "fitness_train": e.get("fitness_train", 0),
                    "generation": e.get("generation", 0),
                    "fold": e.get("fold", "unknown"),
                    "source": "bee",
                })
    except Exception as ex:
        print(f"Bee extraction error: {ex}")
    return all_entries

def main():
    print("=== Extracting TheHive champions ===")
    hive = extract_hive_champions()
    print(f"  TheHive: {len(hive)} entries")
    print("=== Extracting Bee champions ===")
    bee = extract_bee_champions()
    print(f"  Bee: {len(bee)} entries")
    all_entries = hive + bee
    print(f"\nTotal: {len(all_entries)} entries")

    filtered = [e for e in all_entries
                if e["nb_trades"] >= 3
                and e["drawdown"] < 5.0
                and e["champion_path"]
                and e["fitness"] > -100]
    print(f"Filtered (trades>=3, dd<5%): {len(filtered)} entries")

    filtered.sort(key=lambda e: e["fitness"], reverse=True)
    top20 = filtered[:20]
    print(f"\n=== TOP 20 CHAMPIONS ===")
    for i, e in enumerate(top20):
        print(f"{i+1:2d}. {e['symbol']:12s} | {e['run_id']:40s} | fitness={e['fitness']:+.4f} | np={e['net_profit']:+.2f}% | dd={e['drawdown']:.2f}% | wr={e['win_rate']:.1f}% | pf={e['profit_factor']:.2f} | trades={int(e['nb_trades'])} | {e['source']}")

    output = {
        "top20": top20,
        "total_entries": len(all_entries),
        "filtered_entries": len(filtered),
        "timestamp": subprocess.run(["date","+%Y-%m-%dT%H:%M:%S"], capture_output=True, text=True).stdout.strip(),
        "symbols_in_top20": list(set(e["symbol"] for e in top20)),
    }
    out_path = "/home/aza/projects/jepa_eva/data/champions_selection.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
