import numpy as np
import pandas as pd

import numpy as np
import pandas as pd


def calculate_volume_transition_features(
    df: pd.DataFrame,
    volume_window: int = 20,
    zscore_window: int = 50,
    horizons=(1, 3, 5),
    smoothing: float = 1.0,
):
    """
    Volume + Price State Transition Features

    Input:
        df: OHLCV DataFrame
            columns:
                open
                high
                low
                close
                volume

    Output:
        dict containing:
            - current state
            - transition probabilities
            - multi-step probabilities
            - persistence
            - expected returns
            - expected volume pressure
            - transition risk
            - processed dataframe
    """

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    data = (
        df[required]
        .copy()
        .sort_index()
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna()

    if len(data) < zscore_window + max(horizons) + 20:
        raise ValueError(
            "Not enough OHLCV data"
        )

    # =========================================================
    # 1. BASIC PRICE FEATURES
    # =========================================================

    data["return_1"] = (
        data["close"]
        .pct_change()
    )

    data["return_3"] = (
        data["close"]
        .pct_change(3)
    )

    data["return_5"] = (
        data["close"]
        .pct_change(5)
    )

    candle_range = (
        data["high"] -
        data["low"]
    )

    candle_range = candle_range.replace(
        0,
        np.nan
    )

    data["body"] = (
        data["close"] -
        data["open"]
    )

    data["body_ratio"] = (
        data["body"].abs() /
        candle_range
    )

    data["close_position"] = (
        data["close"] -
        data["low"]
    ) / candle_range

    # =========================================================
    # 2. VOLUME FEATURES
    # =========================================================

    data["volume_ma"] = (
        data["volume"]
        .rolling(volume_window)
        .mean()
    )

    data["volume_std"] = (
        data["volume"]
        .rolling(zscore_window)
        .std()
    )

    data["volume_ratio"] = (
        data["volume"] /
        data["volume_ma"]
    )

    data["volume_zscore"] = (
        (
            data["volume"] -
            data["volume"]
            .rolling(zscore_window)
            .mean()
        )
        /
        data["volume_std"]
    )

    data["volume_change"] = (
        data["volume"]
        .pct_change()
    )

    # =========================================================
    # 3. SIGNED VOLUME
    # =========================================================

    direction = np.sign(
        data["close"] -
        data["open"]
    )

    data["signed_volume"] = (
        data["volume"] *
        direction
    )

    data["volume_pressure"] = (
        data["signed_volume"]
        .rolling(volume_window)
        .sum()
        /
        data["volume"]
        .rolling(volume_window)
        .sum()
    )

    # =========================================================
    # 4. BUY / SELL SCORE
    # =========================================================

    # Volume strength
    volume_strength = (
        data["volume_ratio"]
        .clip(0, 3)
        / 3
    )

    # Candle strength
    candle_strength = (
        data["body_ratio"]
        .clip(0, 1)
    )

    # Direction
    price_direction = np.sign(
        data["close"] -
        data["open"]
    )

    # Pressure
    pressure = (
        data["volume_pressure"]
        .clip(-1, 1)
    )

    # Composite score [-1, +1]
    data["volume_signal"] = (
        0.40 *
        price_direction *
        volume_strength

        +

        0.25 *
        price_direction *
        candle_strength

        +

        0.35 *
        pressure
    )

    # =========================================================
    # 5. STATE CLASSIFICATION
    # =========================================================

    # Five states:
    #
    # +2 Strong Buy
    # +1 Buy
    #  0 Neutral
    # -1 Sell
    # -2 Strong Sell

    conditions = [
        data["volume_signal"] >= 0.65,

        data["volume_signal"] >= 0.20,

        data["volume_signal"] <= -0.65,

        data["volume_signal"] <= -0.20,
    ]

    choices = [
        2,
        1,
        -2,
        -1,
    ]

    data["state"] = np.select(
        conditions,
        choices,
        default=0
    )

    # =========================================================
    # 6. STATE LABEL
    # =========================================================

    state_labels = {
        2: "STRONG_BUY",
        1: "BUY",
        0: "NEUTRAL",
        -1: "SELL",
        -2: "STRONG_SELL",
    }

    data["state_label"] = (
        data["state"]
        .map(state_labels)
    )

    states = [
        2,
        1,
        0,
        -1,
        -2,
    ]

    # =========================================================
    # 7. FUTURE RETURNS
    # =========================================================

    for h in horizons:

        data[f"future_return_{h}"] = (
            data["close"]
            .shift(-h)
            /
            data["close"]
            - 1
        )

    # =========================================================
    # 8. NEXT STATE
    # =========================================================

    data["next_state"] = (
        data["state"]
        .shift(-1)
    )

    transitions = (
        data[
            [
                "state",
                "next_state"
            ]
        ]
        .dropna()
        .copy()
    )

    transitions["state"] = (
        transitions["state"]
        .astype(int)
    )

    transitions["next_state"] = (
        transitions["next_state"]
        .astype(int)
    )

    # =========================================================
    # 9. TRANSITION COUNTS
    # =========================================================

    transition_counts = pd.crosstab(
        transitions["state"],
        transitions["next_state"]
    )

    transition_counts = (
        transition_counts
        .reindex(
            index=states,
            columns=states,
            fill_value=0
        )
    )

    # =========================================================
    # 10. LAPLACE SMOOTHING
    # =========================================================

    smoothed_counts = (
        transition_counts +
        smoothing
    )

    transition_probability = (
        smoothed_counts
        .div(
            smoothed_counts.sum(axis=1),
            axis=0
        )
    )

    # =========================================================
    # 11. BASIC TRANSITION PROBABILITIES
    # =========================================================

    p_buy_to_sell = (
        transition_probability
        .loc[1, -1]
    )

    p_sell_to_buy = (
        transition_probability
        .loc[-1, 1]
    )

    p_buy_continue = (
        transition_probability
        .loc[1, 1]
    )

    p_sell_continue = (
        transition_probability
        .loc[-1, -1]
    )

    # =========================================================
    # 12. STRONG TRANSITIONS
    # =========================================================

    p_strong_buy_continue = (
        transition_probability
        .loc[2, 2]
    )

    p_strong_sell_continue = (
        transition_probability
        .loc[-2, -2]
    )

    p_strong_buy_to_sell = (
        transition_probability
        .loc[2, -2]
        +
        transition_probability
        .loc[2, -1]
    )

    p_strong_sell_to_buy = (
        transition_probability
        .loc[-2, 2]
        +
        transition_probability
        .loc[-2, 1]
    )

    # =========================================================
    # 13. MULTI-STEP TRANSITION PROBABILITY
    # =========================================================

    transition_matrix = (
        transition_probability
        .values
    )

    multi_step_probabilities = {}

    for h in horizons:

        matrix_h = np.linalg.matrix_power(
            transition_matrix,
            h
        )

        multi_step_probabilities[h] = (
            pd.DataFrame(
                matrix_h,
                index=states,
                columns=states
            )
        )

    # =========================================================
    # 14. CURRENT STATE
    # =========================================================

    current_state = int(
        data["state"].iloc[-1]
    )

    current_state_label = (
        state_labels[current_state]
    )

    current_transition = (
        transition_probability
        .loc[current_state]
    )

    # =========================================================
    # 15. EXPECTED RETURN BY STATE
    # =========================================================

    expected_returns = {}

    for h in horizons:

        expected_returns[h] = (
            data
            .groupby("state")
            [f"future_return_{h}"]
            .mean()
            .reindex(states)
        )

    # =========================================================
    # 16. EXPECTED RETURN BY TRANSITION
    # =========================================================

    transition_expected_returns = {}

    for h in horizons:

        temp = data[
            [
                "state",
                "next_state",
                f"future_return_{h}"
            ]
        ].dropna()

        transition_expected_returns[h] = (
            temp
            .groupby(
                [
                    "state",
                    "next_state"
                ]
            )[f"future_return_{h}"]
            .mean()
            .unstack()
            .reindex(
                index=states,
                columns=states
            )
        )

    # =========================================================
    # 17. EXPECTED VOLUME PRESSURE BY STATE
    # =========================================================

    expected_pressure = (
        data
        .groupby("state")
        ["volume_pressure"]
        .mean()
        .reindex(states)
    )

    expected_volume_ratio = (
        data
        .groupby("state")
        ["volume_ratio"]
        .mean()
        .reindex(states)
    )

    # =========================================================
    # 18. PERSISTENCE
    # =========================================================

    def calculate_run_lengths(series, state):

        mask = series == state

        groups = (
            mask.ne(mask.shift())
            .cumsum()
        )

        runs = (
            mask
            .groupby(groups)
            .sum()
        )

        return runs[runs > 0]

    persistence = {}

    for state in states:

        runs = calculate_run_lengths(
            data["state"],
            state
        )

        persistence[state] = {
            "mean": runs.mean()
            if len(runs)
            else np.nan,

            "median": runs.median()
            if len(runs)
            else np.nan,

            "max": runs.max()
            if len(runs)
            else np.nan,

            "count": len(runs),
        }

    # =========================================================
    # 19. TRANSITION RISK
    # =========================================================

    # Current Buy → Sell risk
    if current_state in [1, 2]:

        transition_risk = (
            current_transition[-1] +
            current_transition[-2]
        )

    # Current Sell → Buy risk
    elif current_state in [-1, -2]:

        transition_risk = (
            current_transition[1] +
            current_transition[2]
        )

    else:

        transition_risk = np.nan

    # =========================================================
    # 20. BULL / BEAR PROBABILITY
    # =========================================================

    current_buy_probability = (
        current_transition[2] +
        current_transition[1]
    )

    current_sell_probability = (
        current_transition[-2] +
        current_transition[-1]
    )

    current_neutral_probability = (
        current_transition[0]
    )

    # =========================================================
    # 21. EXPECTED NEXT RETURN
    # =========================================================

    expected_next_return = (
        expected_returns[1]
        .get(
            current_state,
            np.nan
        )
    )

    expected_return_3 = (
        expected_returns[3]
        .get(
            current_state,
            np.nan
        )
        if 3 in horizons
        else np.nan
    )

    expected_return_5 = (
        expected_returns[5]
        .get(
            current_state,
            np.nan
        )
        if 5 in horizons
        else np.nan
    )

    # =========================================================
    # 22. EXPECTED RETURN FROM TRANSITION
    # =========================================================

    transition_er_1 = (
        transition_expected_returns[1]
        if 1 in horizons
        else None
    )

    # =========================================================
    # 23. FEATURE DICT
    # =========================================================

    features = {

        # -------------------------
        # Current regime
        # -------------------------

        "volume_state":
            current_state,

        "volume_state_label":
            current_state_label,

        "volume_signal":
            float(
                data["volume_signal"].iloc[-1]
            ),

        # -------------------------
        # Raw volume
        # -------------------------

        "volume_ratio":
            float(
                data["volume_ratio"].iloc[-1]
            ),

        "volume_zscore":
            float(
                data["volume_zscore"].iloc[-1]
            ),

        "volume_pressure":
            float(
                data["volume_pressure"].iloc[-1]
            ),

        "volume_change":
            float(
                data["volume_change"].iloc[-1]
            ),

        # -------------------------
        # Main transitions
        # -------------------------

        "p_buy_to_sell":
            float(p_buy_to_sell),

        "p_sell_to_buy":
            float(p_sell_to_buy),

        "p_buy_continue":
            float(p_buy_continue),

        "p_sell_continue":
            float(p_sell_continue),

        # -------------------------
        # Strong transitions
        # -------------------------

        "p_strong_buy_continue":
            float(p_strong_buy_continue),

        "p_strong_sell_continue":
            float(p_strong_sell_continue),

        "p_strong_buy_to_sell":
            float(p_strong_buy_to_sell),

        "p_strong_sell_to_buy":
            float(p_strong_sell_to_buy),

        # -------------------------
        # Current state
        # -------------------------

        "p_current_buy":
            float(current_buy_probability),

        "p_current_sell":
            float(current_sell_probability),

        "p_current_neutral":
            float(current_neutral_probability),

        "transition_risk":
            float(transition_risk)
            if not np.isnan(transition_risk)
            else np.nan,

        # -------------------------
        # Expected return
        # -------------------------

        "expected_return_1":
            float(expected_next_return),

        "expected_return_3":
            float(expected_return_3),

        "expected_return_5":
            float(expected_return_5),

        # -------------------------
        # Matrices
        # -------------------------

        #"transition_probability":
         #   transition_probability,

        #"transition_counts":
         #   transition_counts,

        #"multi_step_probabilities":
           # multi_step_probabilities,

        #"expected_returns":
            #expected_returns,

        #"transition_expected_returns":
           # transition_expected_returns,

        #"expected_pressure":
            #expected_pressure,

        #"expected_volume_ratio":
            #expected_volume_ratio,

        #"persistence":
            #persistence,

        # -------------------------
        # Processed OHLCV
        # -------------------------

        #"data":
        #    data,
    }

    return features
