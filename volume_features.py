import numpy as np
import pandas as pd


def calculate_volume_transition_features(
    df,
    volume_window=20,
    zscore_window=50,
    return_horizon=1,
    neutral_threshold=0.5,
):
    """
    Calculate volume-based state transition features from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing:
        open, high, low, close, volume

    volume_window : int
        Window for volume moving average.

    zscore_window : int
        Window for volume z-score.

    return_horizon : int
        Number of candles used to measure future return.

    neutral_threshold : float
        Threshold for classifying volume pressure.

    Returns
    -------
    dict
        Transition probabilities, persistence,
        expected returns and expected volume statistics.
    """

    required = ["open", "high", "low", "close", "volume"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = df[required].copy().dropna()

    if len(data) < max(volume_window, zscore_window) + return_horizon + 10:
        raise ValueError("Not enough OHLCV data")

    # ---------------------------------------------------------
    # 1. Basic volume features
    # ---------------------------------------------------------

    data["volume_ma"] = (
        data["volume"]
        .rolling(volume_window)
        .mean()
    )

    data["volume_ratio"] = (
        data["volume"] /
        data["volume_ma"]
    )

    volume_mean = (
        data["volume"]
        .rolling(zscore_window)
        .mean()
    )

    volume_std = (
        data["volume"]
        .rolling(zscore_window)
        .std()
    )

    data["volume_zscore"] = (
        (data["volume"] - volume_mean) /
        volume_std.replace(0, np.nan)
    )

    data["volume_change"] = (
        data["volume"].pct_change()
    )

    # ---------------------------------------------------------
    # 2. Price movement
    # ---------------------------------------------------------

    data["return"] = data["close"].pct_change()

    data["candle_return"] = (
        data["close"] / data["open"] - 1
    )

    # Body relative to range
    candle_range = (
        data["high"] - data["low"]
    ).replace(0, np.nan)

    data["body_ratio"] = (
        (data["close"] - data["open"]).abs()
        / candle_range
    )

    # ---------------------------------------------------------
    # 3. Signed volume
    # ---------------------------------------------------------

    direction = np.sign(
        data["close"] - data["open"]
    )

    data["signed_volume"] = (
        data["volume"] * direction
    )

    # Smoothed buying/selling pressure
    data["volume_pressure"] = (
        data["signed_volume"]
        .rolling(volume_window)
        .sum()
        /
        data["volume"]
        .rolling(volume_window)
        .sum()
    )

    # ---------------------------------------------------------
    # 4. Classify each candle into Buy / Sell / Neutral
    # ---------------------------------------------------------

    # Strong volume condition
    high_volume = (
        data["volume_ratio"] >=
        (1.0 + neutral_threshold * 0.5)
    )

    low_volume = (
        data["volume_ratio"] <
        (1.0 - neutral_threshold * 0.5)
    )

    bullish = data["close"] > data["open"]
    bearish = data["close"] < data["open"]

    data["state"] = "N"

    data.loc[
        high_volume & bullish,
        "state"
    ] = "B"

    data.loc[
        high_volume & bearish,
        "state"
    ] = "S"

    # ---------------------------------------------------------
    # 5. Remove Neutral states for direct transitions?
    #
    # Keep Neutral in the raw transition matrix because
    # it contains useful information.
    # ---------------------------------------------------------

    data["next_state"] = data["state"].shift(-1)

    transitions = data.dropna(
        subset=["state", "next_state"]
    )

    # ---------------------------------------------------------
    # 6. Transition matrix
    # ---------------------------------------------------------

    transition_counts = pd.crosstab(
        transitions["state"],
        transitions["next_state"]
    )

    states = ["B", "S", "N"]

    transition_counts = transition_counts.reindex(
        index=states,
        columns=states,
        fill_value=0
    )

    transition_probability = (
        transition_counts
        .div(
            transition_counts.sum(axis=1)
            .replace(0, np.nan),
            axis=0
        )
    )

    # ---------------------------------------------------------
    # 7. Main probabilities
    # ---------------------------------------------------------

    p_buy_to_sell = (
        transition_probability.loc["B", "S"]
    )

    p_sell_to_buy = (
        transition_probability.loc["S", "B"]
    )

    p_buy_continue = (
        transition_probability.loc["B", "B"]
    )

    p_sell_continue = (
        transition_probability.loc["S", "S"]
    )

    # ---------------------------------------------------------
    # 8. Future return
    # ---------------------------------------------------------

    data["future_return"] = (
        data["close"]
        .shift(-return_horizon)
        / data["close"]
        - 1
    )

    # ---------------------------------------------------------
    # 9. Expected return conditional on state
    # ---------------------------------------------------------

    expected_return_by_state = (
        data.groupby("state")["future_return"]
        .mean()
        .reindex(states)
    )

    # ---------------------------------------------------------
    # 10. Expected volume change conditional on state
    # ---------------------------------------------------------

    expected_volume_change = (
        data.groupby("state")["volume_change"]
        .mean()
        .reindex(states)
    )

    # ---------------------------------------------------------
    # 11. Expected volume ratio
    # ---------------------------------------------------------

    expected_volume_ratio = (
        data.groupby("state")["volume_ratio"]
        .mean()
        .reindex(states)
    )

    # ---------------------------------------------------------
    # 12. Expected pressure
    # ---------------------------------------------------------

    expected_pressure = (
        data.groupby("state")["volume_pressure"]
        .mean()
        .reindex(states)
    )

    # ---------------------------------------------------------
    # 13. Transition-specific expected returns
    # ---------------------------------------------------------

    transitions["future_return"] = (
        data.loc[
            transitions.index,
            "future_return"
        ]
    )

    transition_expected_return = (
        transitions
        .groupby(["state", "next_state"])["future_return"]
        .mean()
        .unstack()
        .reindex(index=states, columns=states)
    )

    # ---------------------------------------------------------
    # 14. Transition-specific expected volume ratio
    # ---------------------------------------------------------

    transitions["volume_ratio"] = (
        data.loc[
            transitions.index,
            "volume_ratio"
        ]
    )

    transition_expected_volume = (
        transitions
        .groupby(["state", "next_state"])["volume_ratio"]
        .mean()
        .unstack()
        .reindex(index=states, columns=states)
    )

    # ---------------------------------------------------------
    # 15. Persistence
    # ---------------------------------------------------------

    buy_mask = data["state"] == "B"
    sell_mask = data["state"] == "S"

    buy_runs = (
        buy_mask
        .groupby(
            (~buy_mask).cumsum()
        )
        .sum()
    )

    sell_runs = (
        sell_mask
        .groupby(
            (~sell_mask).cumsum()
        )
        .sum()
    )

    buy_runs = buy_runs[buy_runs > 0]
    sell_runs = sell_runs[sell_runs > 0]

    expected_buy_duration = (
        buy_runs.mean()
        if len(buy_runs)
        else np.nan
    )

    expected_sell_duration = (
        sell_runs.mean()
        if len(sell_runs)
        else np.nan
    )

    # ---------------------------------------------------------
    # 16. Current state
    # ---------------------------------------------------------

    current_state = data["state"].iloc[-1]

    # ---------------------------------------------------------
    # 17. Current-state probabilities
    # ---------------------------------------------------------

    if current_state in states:
        current_transition_prob = (
            transition_probability.loc[current_state]
        )
    else:
        current_transition_prob = pd.Series(
            np.nan,
            index=states
        )

    # ---------------------------------------------------------
    # 18. Expected next return based on current state
    # ---------------------------------------------------------

    expected_next_return = (
        expected_return_by_state
        .get(current_state, np.nan)
    )

    # ---------------------------------------------------------
    # 19. Build output
    # ---------------------------------------------------------

    features = {

        # Current state
        "current_volume_state": current_state,

        # Transition probabilities
        "p_buy_to_sell": float(p_buy_to_sell),
        "p_sell_to_buy": float(p_sell_to_buy),

        "p_buy_continue": float(p_buy_continue),
        "p_sell_continue": float(p_sell_continue),

        # Current-state transition probabilities
        "p_current_to_buy": float(
            current_transition_prob.get("B", np.nan)
        ),

        "p_current_to_sell": float(
            current_transition_prob.get("S", np.nan)
        ),

        "p_current_to_neutral": float(
            current_transition_prob.get("N", np.nan)
        ),

        # Expected returns
        "expected_return_buy": float(
            expected_return_by_state["B"]
        ),

        "expected_return_sell": float(
            expected_return_by_state["S"]
        ),

        "expected_return_neutral": float(
            expected_return_by_state["N"]
        ),

        "expected_next_return": float(
            expected_next_return
        ),

        # Expected volume
        "expected_volume_change_buy": float(
            expected_volume_change["B"]
        ),

        "expected_volume_change_sell": float(
            expected_volume_change["S"]
        ),

        "expected_volume_ratio_buy": float(
            expected_volume_ratio["B"]
        ),

        "expected_volume_ratio_sell": float(
            expected_volume_ratio["S"]
        ),

        # Pressure
        "expected_pressure_buy": float(
            expected_pressure["B"]
        ),

        "expected_pressure_sell": float(
            expected_pressure["S"]
        ),

        # Persistence
        "expected_buy_duration": float(
            expected_buy_duration
        ),

        "expected_sell_duration": float(
            expected_sell_duration
        ),

        # Current raw features
        "volume_ratio": float(
            data["volume_ratio"].iloc[-1]
        ),

        "volume_zscore": float(
            data["volume_zscore"].iloc[-1]
        ),

        "volume_pressure": float(
            data["volume_pressure"].iloc[-1]
        ),

        "volume_change": float(
            data["volume_change"].iloc[-1]
        ),

        # Transition matrices
        "transition_probability": transition_probability,
        "transition_counts": transition_counts,

        "transition_expected_return":
            transition_expected_return,

        "transition_expected_volume":
            transition_expected_volume,

        # Processed dataframe
        #"data": data
    }

    return features
