#!/usr/bin/env python3
import json, urllib.request, time, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
BRIDGE = "http://192.168.1.6:8765"

# Compte
acct = json.loads(urllib.request.urlopen(urllib.request.Request(BRIDGE + "/account"), timeout=10).read())
capital = acct.get("equity", 9569)
log.info(f"Compte: {acct.get('login')} | Equity: {capital}$")

# Champions FTMO
sel = json.loads(open("/home/aza/projects/jepa_eva/data/champions_selection.json").read())
ftmo = [c for c in sel["top20"] if c["symbol"] in ["EURUSD","GBPUSD","USDJPY","US30.cash","US100.cash","GER40.cash","XAUUSD"] and c.get("net_profit",0) > 0]
log.info(f"Champions deployables: {len(ftmo)}")

# Deployer les 3 meilleurs
deployed = []
for ch in ftmo[:3]:
    sym = ch["symbol"]
    vol = max(0.01, min(1.0, round(capital * 0.02 / 10000, 2)))
    log.info(f"Trade {ch['run_id']} | {sym} | vol={vol} | np={ch.get('net_profit',0):.1f}%")
    order = json.dumps({"symbol": sym, "volume": vol, "type": "buy", "sl": 50, "tp": 100}).encode()
    req = urllib.request.Request(BRIDGE + "/trade", order, {"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        log.info(f"  -> {resp}")
        deployed.append(resp)
    except Exception as e:
        log.error(f"  -> ECHEC: {e}")
    time.sleep(2)

# Vérifier
if deployed:
    pos = json.loads(urllib.request.urlopen(urllib.request.Request(BRIDGE + "/positions"), timeout=10).read())
    log.info(f"Positions ouvertes: {pos.get('count', 0)}")
    log.info(f"Equity: {json.loads(urllib.request.urlopen(urllib.request.Request(BRIDGE + '/account'), timeout=10).read()).get('equity', '?')}$")
log.info("DEPLOIEMENT TERMINE")
