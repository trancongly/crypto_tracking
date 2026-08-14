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

    # Không có dữ liệu
    if df.empty:
        return pd.DataFrame(
            columns=[
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

    # Chọn các cột OHLCV
    df = df[[
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]].copy()

    # Đưa index Timestamp thành column
    df = df.reset_index()

    # Đổi tên cột
    df.columns = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    # Đảm bảo các cột OHLCV là numeric
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Loại bỏ row có NaN trong OHLCV
    df = df.dropna(
        subset=numeric_cols
    )

    # Loại bỏ row có volumdde = 0
    df = df[df["volume"] > 0]

    # Reset index
    df = df.reset_index(drop=True)

    return df
