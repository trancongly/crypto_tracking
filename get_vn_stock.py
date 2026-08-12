import yfinance as yf
import pandas as pd

def get_ohlcv_vn(symbol, interval="1d", period="1y"):
    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return pd.DataFrame(
            columns=["time", "open", "high", "low", "close", "volume"]
        )

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.reset_index()

    df.columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    # Convert Timestamp -> Unix milliseconds
    #df["time"] = (
     #   pd.to_datetime(df["time"])
      #  .astype("int64") // 10**6
    #)

    return df

# Daily
#daily = get_ohlcv_vn("FPT.VN", "1d", "1mo")
#weekly = get_ohlcv_vn("FPT.VN", "1wk", "3y")

#print(daily["low"])
#print(weekly)
#print(daily.columns[0])
