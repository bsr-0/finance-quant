"""Verify the double-adjusted-split repair landed correctly.

Read-only. Checks a known reference series and counts remaining discontinuities.
"""

from __future__ import annotations

import duckdb

REFERENCE_DATES = (
    "2010-01-04",
    "2015-01-02",
    "2020-01-02",
    "2020-08-28",
    "2020-08-31",
    "2026-05-22",
)


def main() -> None:
    con = duckdb.connect("data/market_data.duckdb", read_only=True)

    placeholders = ", ".join(f"'{d}'" for d in REFERENCE_DATES)
    ref = con.execute(f"""
        SELECT p.date, round(p.close, 3) AS close
        FROM cur_prices_ohlcv_daily p
        JOIN dim_symbol s ON s.symbol_id = p.symbol_id
        WHERE s.ticker = 'AAPL' AND p.date IN ({placeholders})
        ORDER BY p.date
        """).df()
    print("AAPL reference closes (true adjusted: 2010 ~6.4, 2015 ~24, 2026 ~309):")
    print(ref.to_string(index=False))

    remaining = con.execute("""
        WITH px AS (
            SELECT close,
                   LAG(close) OVER (PARTITION BY symbol_id ORDER BY date) AS prev_close
            FROM cur_prices_ohlcv_daily
        )
        SELECT count(*) FROM px WHERE prev_close > 0 AND abs(ln(close / prev_close)) > 0.4
        """).fetchone()[0]
    print(f"\nBars with >49% single-day move remaining: {remaining}")

    # Same predicate the repair uses, including the fit margin.
    still_bad = con.execute("""
        WITH px AS (
            SELECT symbol_id, date, close,
                   LAG(close) OVER (PARTITION BY symbol_id ORDER BY date) AS prev_close
            FROM cur_prices_ohlcv_daily
        )
        SELECT count(*)
        FROM px p
        JOIN cur_corporate_actions ca
          ON ca.symbol_id = p.symbol_id
         AND ca.action_date = p.date
         AND ca.action_type = 'split'
        WHERE p.prev_close > 0 AND p.close > 0 AND ca.ratio > 0
          AND abs(ln(ca.ratio)) >= 0.1
          AND abs(ln(p.close / p.prev_close) - ln(ca.ratio))
              < 0.5 * abs(ln(p.close / p.prev_close))
        """).fetchone()[0]
    print(f"Double-adjusted splits still detected: {still_bad}")

    print("\nLargest remaining single-day moves (should be genuine):")
    biggest = con.execute("""
        WITH px AS (
            SELECT symbol_id, date, close,
                   LAG(close) OVER (PARTITION BY symbol_id ORDER BY date) AS prev_close
            FROM cur_prices_ohlcv_daily
        )
        SELECT s.ticker, p.date, round(p.prev_close, 2) AS prev_close,
               round(p.close, 2) AS close,
               round((p.close / p.prev_close - 1) * 100, 1) AS pct
        FROM px p
        JOIN dim_symbol s ON s.symbol_id = p.symbol_id
        WHERE p.prev_close > 0
        ORDER BY abs(ln(p.close / p.prev_close)) DESC
        LIMIT 8
        """).df()
    print(biggest.to_string(index=False))


if __name__ == "__main__":
    main()
