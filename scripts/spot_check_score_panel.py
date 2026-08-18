"""Gate G2 spot check: does the vectorized score_frame agree with the row-wise
_score_row on real production data, using the exact ticker set and lookback
window the live daily-predictions pipeline uses?

This does NOT diff against data/signals/signals_20260519.csv. That CSV's
underlying rows carry ingested_at timestamps of 2026-05-25 -- six days after
the CSV's own signal date -- so the source data was revised after generation
and the CSV can no longer be reproduced byte-for-bit regardless of scorer
correctness. The meaningful comparison is vectorized vs row-wise on today's
data, which is what this checks.
"""

from __future__ import annotations

import pandas as pd

from pipeline.strategy.price_panel import load_ticker_frames
from pipeline.strategy.signal_panel import score_ticker
from pipeline.strategy.signals import SignalEngine, compute_indicators

SIGNAL_DATE = pd.Timestamp("2026-05-19")
LOOKBACK_DAYS = 252
CSV_PATH = "data/signals/signals_20260519.csv"


def main() -> None:
    csv = pd.read_csv(CSV_PATH)
    tickers = sorted(set(csv["ticker"]) | {"SPY"})

    frames, breaks = load_ticker_frames(tickers, start="2020-01-01", end=SIGNAL_DATE)
    print(f"Loaded {len(frames)}/{len(tickers)} tickers, {len(breaks)} price repairs applied")

    spy_full = frames["SPY"]["close"]
    engine = SignalEngine()

    mismatches = 0
    for ticker in [t for t in tickers if t != "SPY"]:
        if ticker not in frames:
            print(f"  {ticker}: MISSING from loaded frames")
            mismatches += 1
            continue

        df = frames[ticker][frames[ticker].index <= SIGNAL_DATE].tail(LOOKBACK_DAYS)
        spy = spy_full[spy_full.index <= SIGNAL_DATE].tail(LOOKBACK_DAYS)
        if SIGNAL_DATE not in df.index:
            continue

        vectorized = score_ticker(df, engine, spy_prices=spy).loc[SIGNAL_DATE]

        indicators = compute_indicators(df)
        row_wise = engine._score_row(indicators.loc[SIGNAL_DATE])
        total, trend, pb, vol, volat = row_wise

        diffs = {}
        for name, expected in [
            ("score", total),
            ("trend_pts", trend),
            ("pullback_pts", pb),
            ("volume_pts", vol),
            ("volatility_pts", volat),
        ]:
            if int(vectorized[name]) != int(expected):
                diffs[name] = (int(vectorized[name]), int(expected))
        if diffs:
            print(f"  {ticker}: vectorized vs row-wise MISMATCH {diffs}")
            mismatches += 1

    scored_tickers = len(tickers) - 1  # exclude SPY
    print(f"\n{scored_tickers - mismatches}/{scored_tickers} tickers: vectorized == row-wise")


if __name__ == "__main__":
    main()
