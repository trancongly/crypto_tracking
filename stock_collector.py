import requests
import sys
import pandas as pd
import pandas_ta as ta
from datetime import datetime, UTC
import time
from pathlib import Path

from atr_ind import calculate_atr_metrics

from config import *
from get_vn_stock import get_ohlcv_vn

from rsi_features import ml_divergence
from rsi_features import calculate_rsi_features

from stock_price_trend_analyzer import calculate_market_structure

from bb_analyzer import BollingerStructureAnalyzer

from google import genai
try:                                                        client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    client = None


msg = ""
symbol = "FPT.VN"

symbol = sys.argv[1]

folder = Path("logs") / symbol
folder.mkdir(parents=True, exist_ok=True)
filename = folder / f"{datetime.now():%Y%m%d_%H%M%S}.txt"

symbol = symbol + ".VN"
print(symbol)


df_1w = get_ohlcv_vn(symbol, "1wk", "3y")
df_1d = get_ohlcv_vn(symbol, "1d", "1y")

rsi_1w = ta.rsi(df_1w["close"], length=6)
rsi_1d = ta.rsi(df_1d["close"], length=6)
market_features = calculate_market_structure(df=df_1w, rsi=rsi_1w, lookback=50)

#print(market_features)
with open(filename, "a", encoding="utf-8") as f:
    f.write("week data")
    f.write("\n")
    f.write(str(market_features))
    f.write("\n")


analyzer = BollingerStructureAnalyzer()
bb_structures = analyzer.analyze(df_1d.tail(90))
#print(bb_structures)

ml_rsi4 = ml_divergence(df_1d["close"], rsi_1d)

atr_metrics = calculate_atr_metrics(df_1d)

rsi_features = calculate_rsi_features(rsi_1d)
#print(rsi_features)

with open(filename, "a", encoding="utf-8") as f:
    f.write("day data")
    f.write("\n")
    f.write(str(bb_structures))
    f.write("\n")
    f.write(str(atr_metrics))
    f.write("\n")
    f.write(str(rsi_features))
