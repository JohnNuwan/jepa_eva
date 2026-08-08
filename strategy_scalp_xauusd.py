#!/usr/bin/env python3
"""EVA-ENSEMBLE-SCALP - 3 agents votent"""
import numpy as np, pandas as pd, json, urllib.request, os, pickle
from datetime import datetime

SYMBOL = "XAUUSD"
TF = "M1"
BRIDGE = "http://192.168.1.6:8765"
COMMENT = "EVA-SCALP"
MAX_POS = 5
LOG = "/home/aza/eva-adam-v2/logs/rl_XAUUSD_scalp.log"

class ScalpAgent:
    def __init__(self, qfile, alpha=0.15, gamma=0.85, eps=0.4):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eps
        self.recent_rewards = []
        self.qfile = qfile
        if os.path.exists(qfile):
            with open(qfile,"rb") as f: self.q_table = pickle.load(f)
    def save(self):
        with open(self.qfile,"wb") as f: pickle.dump(self.q_table, f)
    def get_state(self, df):
        c = df["close"].values; h = df["high"].values; l = df["low"].values; v = df["volume"].values
        if len(c) < 10: return (5,0,0,0,0)
        micro = 1 if c[-1] > c[-3] else -1
        delta = np.diff(c[-8:])
        gain = np.mean(delta[delta>0]) if np.any(delta>0) else 0.001
        loss = -np.mean(delta[delta<0]) if np.any(delta<0) else 0.001
        rsi = 100 - 100/(1+gain/loss)
        vol = int((h[-1]-l[-1])/c[-1]*10000)
        vol_ratio = int(v[-1]/(np.mean(v[-20:])+1e-6)*10) if len(v)>20 else 5
        now = datetime.now()
        is_news = 1 if 17<=now.hour<=19 else 0
        return (int(rsi//10), vol, micro, min(vol_ratio,9), is_news)
    def act(self, state):
        if state not in self.q_table: self.q_table[state]={"buy":0,"sell":0,"hold":0}
        if len(self.recent_rewards)>10:
            wr=sum(1 for r in self.recent_rewards[-10:] if r>0)/10
            self.epsilon=max(0.1, min(0.6, 0.4*(1-wr)*2))
        if np.random.random()<self.epsilon:
            return np.random.choice(["buy","sell","hold"])
        return max(self.q_table[state], key=self.q_table[state].get)
    def get_size(self, state, action):
        if state not in self.q_table: self.q_table[state]={"buy":0,"sell":0,"hold":0}
        vals=list(self.q_table[state].values())
        conf=self.q_table[state][action]-np.median(vals) if vals else 0
        vol=state[1]
        base=np.clip(abs(conf)/0.3, 0.01, 0.03)
        if vol>5: base=min(0.05, base*1.5)
        if vol>10: base=min(0.08, base*2)
        return round(base,2)
    def update(self, s, a, r, ns):
        for st in [s, ns]:
            if st not in self.q_table: self.q_table[st]={"buy":0,"sell":0,"hold":0}
        self.recent_rewards.append(r)
        mf=max(self.q_table[ns].values())
        self.q_table[s][a]+=self.alpha*(r+self.gamma*mf-self.q_table[s][a])
        self.save()

def log(m):
    t=datetime.now().isoformat()[:19]
    with open(LOG,"a") as f: f.write(t+" "+m+chr(10))
    print(t,m)

log("="*40)
log("EVA-ENSEMBLE XAUUSD M1")

now=datetime.now()
if 17<=now.hour<=19:
    log("News period - skip"); exit()

try:
    with urllib.request.urlopen(BRIDGE+"/ohlcv/XAUUSD/50/M1", timeout=10) as r:
        df=pd.DataFrame(json.loads(r.read().decode())["bars"])
except Exception as e:
    log("ERR: "+str(e)); exit()

# Create 3 agents with different params
agents = [
    ScalpAgent("/home/aza/eva-adam-v2/data/rl_qtable_XAUUSD_scalp.pkl", 0.15, 0.85, 0.4),
    ScalpAgent("/home/aza/eva-adam-v2/data/rl_qtable_XAUUSD_scalp_2.pkl", 0.2, 0.8, 0.5),
    ScalpAgent("/home/aza/eva-adam-v2/data/rl_qtable_XAUUSD_scalp_3.pkl", 0.1, 0.9, 0.3),
]

state = agents[0].get_state(df)
votes = []
for i, ag in enumerate(agents):
    action = ag.act(state)
    votes.append(action)
    log(f"  Agent {i+1}: {action} (eps={ag.epsilon:.2f})")

# Ensemble vote: majority wins
from collections import Counter
vote_count = Counter(votes)
winner = vote_count.most_common(1)[0][0]
log(f"Vote: {dict(vote_count)} -> {winner}")

# If tie, use agent 0 (most experienced)
if vote_count.most_common(1)[0][1] == vote_count.most_common(2)[0][1]:
    winner = agents[0].act(state)
    log(f"Tie broken by agent 0: {winner}")

try:
    with urllib.request.urlopen(BRIDGE+"/positions", timeout=5) as r:
        pos=json.loads(r.read().decode()).get("positions",[])
        my_pos=[p for p in pos if COMMENT in p.get("comment","")]
        existing=len(my_pos)
except Exception: existing=0

size = max(ag.get_size(state, winner) for ag in agents)
log(f"Ensemble: {winner} size={size} pos={existing}")

if winner in ["buy","sell"] and existing < MAX_POS:
    order={"symbol":SYMBOL,"volume":size,"type":winner,"comment":COMMENT}
    try:
        req=urllib.request.Request(BRIDGE+"/trade", data=json.dumps(order).encode(), headers={"Content-Type":"application/json"})
        resp=json.loads(urllib.request.urlopen(req, timeout=5).read().decode())
        log("ORDER: "+str(resp))
    except Exception as e:
        log("ERR: "+str(e))

# Reward all agents
try:
    with urllib.request.urlopen(BRIDGE+"/positions", timeout=5) as r:
        for p in json.loads(r.read().decode()).get("positions",[]):
            if COMMENT in p.get("comment","") and p.get("profit",0)!=0:
                reward=float(p["profit"])/max(np.mean(df["high"].values[-7:]-df["low"].values[-7:]), 0.01)
                for ag in agents:
                    ag.update(state, winner, reward, state)
except Exception: pass
log("done")
