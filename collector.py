import sqlite3
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, UTC
import time
from atr_ind import calculate_atr_metrics

from config import *

from rsi_features import ml_divergence
from rsi_features import calculate_rsi_features

from price_trend_analyzer import calculate_market_structure

from bb_analyzer import BollingerStructureAnalyzer

from google import genai
try:                                                        client = genai.Client(api_key=GEMINI_API_KEY)
except Exception:
    client = None

msg = ""

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS snapshots(
id INTEGER PRIMARY KEY,
timestamp TEXT,
symbol TEXT,
timeframe TEXT,
price REAL,
rsi REAL,
volume REAL,
bb_upper REAL,
bb_mid REAL,
bb_lower REAL
)
""")

def get_data(symbol, interval):

    url = "https://api.binance.com/api/v3/klines"

    r = requests.get(
        url,
        params={
            "symbol": symbol,
            "interval": interval,
            "limit": 200
        }
    )

    df = pd.DataFrame(r.json())

    df = df.iloc[:, :6]

    df.columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    #print(df)
    return df


for symbol in SYMBOLS:

    for tf in TIMEFRAMES:

        df = get_data(symbol, tf)

        rsi = ta.rsi(df["close"], length=6)
        # Market structure only on 1D
        if tf == "1d":
            market_features = calculate_market_structure(
            df=df,
            rsi=rsi,
            lookback=90
        )

            #print("\n=== 1D Market Structure ===")
            #print(market_features)


        if tf == "4h":
            analyzer = BollingerStructureAnalyzer()
            bb_structures = analyzer.analyze(df.tail(90))

            ml_rsi4 = ml_divergence(df["close"], rsi)
            atr_metrics = calculate_atr_metrics(df)

            #print(ml_rsi4)   
            #print("\n=== 4H BB Structure ===")
            #print(bb_structures)

        if tf == "1h":
            ml_rsi = ml_divergence(df["close"], rsi)

            #print(ml_rsi)   
            rsi_features = calculate_rsi_features(rsi)
            #print("\n=== 1H RSI Structure ===")
            #print(rsi_features)

    prompt = f"""
You are an expert crypto trader. Analyze market data for
 {symbol} with 1d, 4h, 1h timeframe: 
 1d: {market_features} 
 4h: {bb_structures}{atr_metrics} 
 1h: {rsi_features}.

 Tasks: Identify 3 most importance resistance and support.              

Return JSON ONLY:
{{
    "resistance": [0, 0, 0],
    "support": [0, 0, 0]
}}
 """

    print(prompt)
    response = client.models.generate_content(                  model="gemini-2.5-flash",
            contents=prompt
    )

    text = response.text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    print(text)

    time.sleep(60)

    msg += f"""                                         {symbol}
{text}
------------------                                      """   


t = requests.post(                                          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",                                                    data={                                                      "chat_id": TELEGRAM_CHAT_ID,                            "text": msg                                         }                                                   )
print(t.status_code)    
print(t.text)

print("Collector finished")
