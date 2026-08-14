from datetime import datetime, timezone, timedelta


def calculate_market_structure(
    df,
    rsi,
    lookback=30,
    timezone_offset=7
):
    """
    Calculate market structure metrics.

    Args:
        df: DataFrame with columns:
            time, open, high, low, close, volume

        rsi: RSI Series aligned with df index

        lookback: Number of candles to analyze

        timezone_offset: Timezone offset in hours

    Returns:
        dict
    """

    recent = df.tail(lookback).copy()
    recent_rsi = rsi.tail(lookback).copy()

    for col in ["open", "high", "low", "close", "volume"]:
        recent[col] = recent[col].astype(float)

    # Find high and low points

    high_idx = recent["high"].idxmax()
    low_idx = recent["low"].idxmin()

    high_price = float(recent.loc[high_idx, "high"])
    low_price = float(recent.loc[low_idx, "low"])

    # Convert timestamps

    tz = timezone(timedelta(hours=timezone_offset))

    high_time = recent.loc[high_idx, "time"]

    low_time = recent.loc[low_idx, "time"]

    # Relative positions

    high_pos = recent.index.get_loc(high_idx)
    low_pos = recent.index.get_loc(low_idx)

    # Current values

    current_close = float(recent["close"].iloc[-1])
    current_volume = float(recent["volume"].iloc[-1])
    current_rsi = float(recent_rsi.iloc[-1])

    # Direction

    if low_pos < high_pos:
        direction = "low_to_high"
    else:
        direction = "high_to_low"

    # Time structure

    bars_between_high_low = abs(high_pos - low_pos)

    days_since_high = len(recent) - 1 - high_pos
    days_since_low = len(recent) - 1 - low_pos

    # Range

    range_pct = (
        (high_price - low_price)
        / low_price
        * 100
    )

    # Current position inside range

    if high_price > low_price:
        price_position_30d = (
            (current_close - low_price)
            / (high_price - low_price)
        )
    else:
        price_position_30d = 0.0

    # Distance metrics

    distance_from_high_pct = (
        (high_price - current_close)
        / high_price
        * 100
    )

    distance_from_low_pct = (
        (current_close - low_price)
        / low_price
        * 100
    )

    # RSI structure

    high_rsi = float(recent_rsi.loc[high_idx])
    low_rsi = float(recent_rsi.loc[low_idx])

    # Volume structure

    high_volume = float(recent.loc[high_idx, "volume"])
    low_volume = float(recent.loc[low_idx, "volume"])

    return {

        # Price structure

        "high_price": round(high_price, 8),
        "high_time": high_time,
        "volume_at_high_price": int(high_volume),
        "high_rsi": round(high_rsi, 2),

        "low_price": round(low_price, 8),
        "low_time": low_time,
        "volume_at_low_price": int(low_volume),
        "low_rsi": round(low_rsi, 2),

        "current_close": round(current_close, 8),

        "direction": direction,

        "bars_between_high_low":
            int(bars_between_high_low),

        "days_since_high":
            int(days_since_high),

        "days_since_low":
            int(days_since_low),

        "range_pct":
            round(float(range_pct), 2),

        "price_position_30day":
            round(float(price_position_30d), 4),

        "distance_from_high_pct":
            round(float(distance_from_high_pct), 2),

        "distance_from_low_pct":
            round(float(distance_from_low_pct), 2),

        # RSI structure
        "current_rsi":
            round(current_rsi, 2),

        # Volume structure
        "current_volume":
            int(current_volume)
    }

###
# Calculate HH/HL/LH/LL
###
def calculate_zigzag(df, threshold_pct=5.0):
    """
    ZigZag based on percentage reversal.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: high, low
    threshold_pct : float
        Reversal threshold in percent.

    Returns
    -------
    pd.DataFrame
        index, price, type
    """

    highs = df["high"].values
    lows = df["low"].values

    pivots = []

    pivot_idx = 0
    pivot_price = highs[0]

    trend = None  # up / down

    for i in range(1, len(df)):

        high = highs[i]
        low = lows[i]

        # Initialization
        if trend is None:

            up_move = (high - pivot_price) / pivot_price * 100
            down_move = (pivot_price - low) / pivot_price * 100

            if up_move >= threshold_pct:
                trend = "up"
                pivot_idx = i
                pivot_price = high

            elif down_move >= threshold_pct:
                trend = "down"
                pivot_idx = i
                pivot_price = low

            continue

        # Uptrend
        if trend == "up":

            # New higher high
            if high > pivot_price:
                pivot_idx = i
                pivot_price = high

            # Reversal
            reversal = (pivot_price - low) / pivot_price * 100

            if reversal >= threshold_pct:
                pivots.append(
                    {
                        "index": pivot_idx,
                        "price": pivot_price,
                        "type": "HIGH"
                    }
                )

                trend = "down"
                pivot_idx = i
                pivot_price = low

        # Downtrend
        else:

            # New lower low
            if low < pivot_price:
                pivot_idx = i
                pivot_price = low

            reversal = (high - pivot_price) / pivot_price * 100

            if reversal >= threshold_pct:
                pivots.append(
                    {
                        "index": pivot_idx,
                        "price": pivot_price,
                        "type": "LOW"
                    }
                )

                trend = "up"
                pivot_idx = i
                pivot_price = high

    # Append last pivot
    if trend == "up":
        pivots.append(
            {
                "index": pivot_idx,
                "price": pivot_price,
                "type": "HIGH"
            }
        )
    elif trend == "down":
        pivots.append(
            {
                "index": pivot_idx,
                "price": pivot_price,
                "type": "LOW"
            }
        )

    return pd.DataFrame(pivots)

def classify_market_structure(zigzag_df):
    """
    Add HH/HL/LH/LL labels.
    """

    labels = []
    last_high = None
    last_low = None

    for _, row in zigzag_df.iterrows():

        if row["type"] == "HIGH":

            if last_high is None:
                labels.append("HIGH")
            elif row["price"] > last_high:
                labels.append("HH")
            else:
                labels.append("LH")

            last_high = row["price"]

        else:

            if last_low is None:
                labels.append("LOW")
            elif row["price"] > last_low:
                labels.append("HL")
            else:
                labels.append("LL")

            last_low = row["price"]

    result = zigzag_df.copy()
    result["label"] = labels

    return result
