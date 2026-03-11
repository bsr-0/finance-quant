"""Portfolio construction utilities for evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SignalPortfolioConfig:
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    max_weight: float = 0.05
    capital: float = 1_000_000


@dataclass
class ProbPortfolioConfig:
    edge_threshold: float = 0.02
    notional_per_trade: float = 1_000
    holding_period_days: int = 7


def generate_positions_from_signals(
    signals: pd.DataFrame,
    config: SignalPortfolioConfig | None = None,
) -> pd.DataFrame:
    """Generate daily positions from cross-sectional signals.

    Expected columns: date, symbol, signal, price. Optional: adv.
    Returns a DataFrame indexed by date with columns = symbols (share positions).
    """
    config = config or SignalPortfolioConfig()
    required = {"date", "symbol", "signal", "price"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = signals.copy()
    df["date"] = pd.to_datetime(df["date"])

    positions = []

    for dt, day in df.groupby("date"):
        day = day.dropna(subset=["signal", "price"])
        if day.empty:
            continue

        n = len(day)
        n_long = max(int(np.floor(n * config.long_quantile)), 1)
        n_short = max(int(np.floor(n * config.short_quantile)), 1)

        long_candidates = day.nlargest(n_long, "signal")
        short_candidates = day.nsmallest(n_short, "signal")

        weights = pd.Series(0.0, index=day["symbol"])
        if not long_candidates.empty:
            long_weight = 1.0 / len(long_candidates)
            weights.loc[long_candidates["symbol"].values] = long_weight
        if not short_candidates.empty:
            short_weight = -1.0 / len(short_candidates)
            weights.loc[short_candidates["symbol"].values] = short_weight

        weights = weights.clip(lower=-config.max_weight, upper=config.max_weight)

        day_prices = day.set_index("symbol")["price"]
        shares = (weights * config.capital / day_prices).replace([np.inf, -np.inf], 0).fillna(0)
        shares.name = dt
        positions.append(shares)

    if not positions:
        return pd.DataFrame()

    positions_df = pd.DataFrame(positions).fillna(0.0).sort_index()
    positions_df.index.name = "date"
    return positions_df


def generate_positions_from_probs(
    probs: pd.DataFrame,
    config: ProbPortfolioConfig | None = None,
) -> pd.DataFrame:
    """Generate positions for prediction markets based on model edge.

    Expected columns: date, contract_id, market_price, model_prob.
    Returns positions in contracts (positive = YES, negative = NO).
    """
    config = config or ProbPortfolioConfig()
    required = {"date", "contract_id", "market_price", "model_prob"}
    missing = required - set(probs.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = probs.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["contract_id", "date"])

    positions = {}

    for contract_id, grp in df.groupby("contract_id"):
        grp = grp.reset_index(drop=True)
        pos = pd.Series(0.0, index=grp["date"])
        open_side = 0
        open_date = None

        for i, row in grp.iterrows():
            edge = row["model_prob"] - row["market_price"]
            dt = row["date"]
            if open_side != 0:
                holding_days = (dt - open_date).days if open_date else 0
                if holding_days >= config.holding_period_days:
                    open_side = 0
                    open_date = None

            if open_side == 0:
                if edge > config.edge_threshold:
                    open_side = 1
                    open_date = dt
                elif edge < -config.edge_threshold:
                    open_side = -1
                    open_date = dt

            pos.loc[dt] = open_side * config.notional_per_trade

        positions[contract_id] = pos

    positions_df = pd.DataFrame(positions).fillna(0.0).sort_index()
    positions_df.index.name = "date"
    return positions_df
