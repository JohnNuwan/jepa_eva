#!/usr/bin/env python3
import sys
sys.path.insert(0, "/home/aza/projects/jepa_eva")
from multi_tf import fetch_ohlcv, _aggregate_h4_from_m30, get_h4_features

# Debug M30 fetch
bars = fetch_ohlcv("US30.cash", 200, "M30")
print("M30 bars:", len(bars))
if bars:
    print("First time:", bars[0]["time"], "close:", bars[0]["close"])
    print("Last time:", bars[-1]["time"], "close:", bars[-1]["close"])

# Debug H4 aggregation
h4 = _aggregate_h4_from_m30(bars)
print("H4 candles:", len(h4))
if h4:
    print("First H4:", h4[0])
    print("Last H4:", h4[-1])

# Debug get_h4_features
result = get_h4_features("US30.cash")
print("get_h4_features:", result)
