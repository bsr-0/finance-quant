"""Load raw data files into raw tables."""

import json
import logging
from pathlib import Path
from uuid import UUID

import pandas as pd
from sqlalchemy import text

from pipeline.db import get_db_manager
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)


class RawLoader:
    """Load raw parquet files into database raw tables."""

    def __init__(self):
        self.db = get_db_manager()
        self._batch_size = get_settings().infrastructure.batch_size

    def _batch_insert(self, conn, insert_sql, records: list[dict]) -> int:
        if not records:
            return 0
        total = 0
        for i in range(0, len(records), self._batch_size):
            batch = records[i : i + self._batch_size]
            conn.execute(insert_sql, batch)
            total += len(batch)
        return total

    def load_fred_observations(
        self,
        file_path: Path,
        run_id: UUID | None = None,
    ) -> int:
        """Load FRED observations from parquet file.

        Persists ``realtime_start`` / ``realtime_end`` from the FRED API so
        that the curated layer can derive correct ``available_time`` (the date
        FRED first published the data point) rather than using the pipeline's
        ``extracted_at`` timestamp.
        """
        logger.info(f"Loading FRED data from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            logger.warning(f"Empty file: {file_path}")
            return 0

        df["run_id"] = run_id

        insert_sql = text("""
            INSERT INTO raw_fred_observations
            (series_code, observation_date, value, realtime_start, realtime_end,
             raw_data, extracted_at, run_id)
            VALUES (:series_code, :date, :value, :realtime_start, :realtime_end,
                    :raw_data, :extracted_at, :run_id)
            ON CONFLICT (series_code, observation_date, realtime_start) DO UPDATE SET
                value = EXCLUDED.value,
                realtime_end = EXCLUDED.realtime_end,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "series_code": row.get("series_code"),
                    "date": row.get("date"),
                    "value": row.get("value"),
                    "realtime_start": row.get("realtime_start"),
                    "realtime_end": row.get("realtime_end"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} FRED observations")
        return rows_loaded

    def load_gdelt_events(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load GDELT events from parquet file."""
        logger.info(f"Loading GDELT data from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            logger.warning(f"Empty file: {file_path}")
            return 0

        insert_sql = text("""
            INSERT INTO raw_gdelt_events
            (gdelt_event_id, event_date, raw_data, extracted_at, run_id)
            VALUES (:event_id, :event_date, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (gdelt_event_id) DO UPDATE SET
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "event_id": row.get("GLOBALEVENTID"),
                    "event_date": row.get("SQLDATE"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} GDELT events")
        return rows_loaded

    def load_polymarket_markets(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Polymarket market metadata from parquet file."""
        logger.info(f"Loading Polymarket markets from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_polymarket_markets
            (venue_market_id, raw_data, extracted_at, run_id)
            VALUES (:market_id, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (venue_market_id) DO UPDATE SET
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            market_id = row.get("id") or row.get("marketId")
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "market_id": market_id,
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} Polymarket markets")
        return rows_loaded

    def load_polymarket_prices(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Polymarket price data from parquet file."""
        logger.info(f"Loading Polymarket prices from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_polymarket_prices
            (venue_market_id, ts, outcome, price, raw_data, extracted_at, run_id)
            VALUES (:market_id, :ts, :outcome, :price, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (venue_market_id, ts, outcome) DO UPDATE SET
                price = EXCLUDED.price,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "market_id": row.get("market_id"),
                    "ts": row.get("timestamp"),
                    "outcome": row.get("outcome") or "YES",
                    "price": row.get("price"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} Polymarket prices")
        return rows_loaded

    def load_polymarket_trades(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Polymarket trades from parquet file."""
        logger.info(f"Loading Polymarket trades from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_polymarket_trades
            (venue_market_id, trade_id, ts, price, size,
             side, raw_data, extracted_at, run_id)
            VALUES (:market_id, :trade_id, :ts, :price,
             :size, :side, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (trade_id) DO UPDATE SET
                price = EXCLUDED.price,
                size = EXCLUDED.size,
                side = EXCLUDED.side,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "market_id": row.get("market_id"),
                    "trade_id": row.get("id") or row.get("trade_id"),
                    "ts": row.get("timestamp"),
                    "price": row.get("price"),
                    "size": row.get("size"),
                    "side": row.get("side"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} Polymarket trades")
        return rows_loaded

    def load_polymarket_orderbooks(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Polymarket orderbook snapshots from parquet file."""
        logger.info(f"Loading Polymarket orderbooks from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_polymarket_orderbook_snapshots
            (venue_market_id, ts, best_bid, best_ask, spread, bids, asks,
             raw_data, extracted_at, run_id)
            VALUES (:market_id, :ts, :best_bid, :best_ask, :spread, :bids, :asks,
                    :raw_data, :extracted_at, :run_id)
            ON CONFLICT (venue_market_id, ts) DO UPDATE SET
                best_bid = EXCLUDED.best_bid,
                best_ask = EXCLUDED.best_ask,
                spread = EXCLUDED.spread,
                bids = EXCLUDED.bids,
                asks = EXCLUDED.asks,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "market_id": row.get("market_id"),
                    "ts": row.get("ts"),
                    "best_bid": row.get("best_bid"),
                    "best_ask": row.get("best_ask"),
                    "spread": row.get("spread"),
                    "bids": row.get("bids"),
                    "asks": row.get("asks"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} Polymarket orderbook snapshots")
        return rows_loaded

    def load_prices_ohlcv(
        self,
        file_path: Path,
        run_id: UUID | None = None,
    ) -> int:
        """Load OHLCV price data from parquet file.

        Also persists ``split_ratio`` and ``dividend`` columns so that the
        curated transform can populate ``cur_corporate_actions``.
        """
        logger.info(f"Loading OHLCV data from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_prices_ohlcv
            (ticker, date, open, high, low, close, adj_close, volume,
             split_ratio, dividend, raw_data, extracted_at, run_id)
            VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume,
                    :split_ratio, :dividend, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (ticker, date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                adj_close = EXCLUDED.adj_close,
                volume = EXCLUDED.volume,
                split_ratio = EXCLUDED.split_ratio,
                dividend = EXCLUDED.dividend,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "ticker": row.get("ticker"),
                    "date": row.get("date"),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "adj_close": row.get("adj_close"),
                    "volume": row.get("volume"),
                    "split_ratio": row.get("split_ratio"),
                    "dividend": row.get("dividend", 0),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} OHLCV records")
        return rows_loaded

    def load_factor_returns(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load factor returns from parquet file."""
        logger.info(f"Loading factor data from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_factor_returns
            (date, mkt_rf, smb, hml, rmw, cma, mom, rf, raw_data, extracted_at, run_id)
            VALUES (:date, :mkt_rf, :smb, :hml, :rmw, :cma, :mom, :rf,
                    :raw_data, :extracted_at, :run_id)
            ON CONFLICT (date) DO UPDATE SET
                mkt_rf = EXCLUDED.mkt_rf,
                smb = EXCLUDED.smb,
                hml = EXCLUDED.hml,
                rmw = EXCLUDED.rmw,
                cma = EXCLUDED.cma,
                mom = EXCLUDED.mom,
                rf = EXCLUDED.rf,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append(
                {
                    "date": row.get("date"),
                    "mkt_rf": row.get("mkt_rf"),
                    "smb": row.get("smb"),
                    "hml": row.get("hml"),
                    "rmw": row.get("rmw"),
                    "cma": row.get("cma"),
                    "mom": row.get("mom"),
                    "rf": row.get("rf"),
                    "raw_data": row.get("raw_data"),
                    "extracted_at": row.get("extracted_at"),
                    "run_id": row.get("run_id"),
                }
            )

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} factor rows")
        return rows_loaded

    def load_sec_fundamentals(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load SEC fundamentals from parquet file."""
        logger.info(f"Loading SEC fundamentals from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_sec_fundamentals
            (ticker, cik, metric_name, metric_label, metric_value, units,
             fiscal_period_end, filing_date, form_type, accession_number,
             fiscal_year, fiscal_period, raw_data, extracted_at, run_id)
            VALUES (:ticker, :cik, :metric_name, :metric_label, :metric_value, :units,
                    :fiscal_period_end, :filing_date, :form_type, :accession_number,
                    :fiscal_year, :fiscal_period, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (ticker, metric_name, fiscal_period_end, form_type, accession_number)
            DO UPDATE SET
                metric_value = EXCLUDED.metric_value,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append({k: row.get(k) for k in [
                "ticker", "cik", "metric_name", "metric_label", "metric_value",
                "units", "fiscal_period_end", "filing_date", "form_type",
                "accession_number", "fiscal_year", "fiscal_period",
                "raw_data", "extracted_at", "run_id",
            ]})

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} SEC fundamentals records")
        return rows_loaded

    def load_sec_insider_trades(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load SEC insider trades from parquet file."""
        logger.info(f"Loading SEC insider trades from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_sec_insider_trades
            (ticker, cik, insider_cik, insider_name, insider_title,
             transaction_date, transaction_type, shares, price_per_share,
             shares_after, ownership_type, accession_number, filing_date,
             raw_data, extracted_at, run_id)
            VALUES (:ticker, :cik, :insider_cik, :insider_name, :insider_title,
                    :transaction_date, :transaction_type, :shares, :price_per_share,
                    :shares_after, :ownership_type, :accession_number, :filing_date,
                    :raw_data, :extracted_at, :run_id)
            ON CONFLICT (accession_number, insider_cik, transaction_date, transaction_type, shares)
            DO UPDATE SET
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append({k: row.get(k) for k in [
                "ticker", "cik", "insider_cik", "insider_name", "insider_title",
                "transaction_date", "transaction_type", "shares", "price_per_share",
                "shares_after", "ownership_type", "accession_number", "filing_date",
                "raw_data", "extracted_at", "run_id",
            ]})

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} SEC insider trades")
        return rows_loaded

    def load_sec_13f_holdings(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load SEC 13F holdings from parquet file."""
        logger.info(f"Loading SEC 13F holdings from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_sec_13f_holdings
            (filer_cik, filer_name, report_date, filing_date, cusip, issuer_name,
             class_title, market_value, shares_held, shares_type, put_call,
             investment_discretion, voting_authority_sole, voting_authority_shared,
             voting_authority_none, accession_number, raw_data, extracted_at, run_id)
            VALUES (:filer_cik, :filer_name, :report_date, :filing_date, :cusip, :issuer_name,
                    :class_title, :market_value, :shares_held, :shares_type, :put_call,
                    :investment_discretion, :voting_authority_sole, :voting_authority_shared,
                    :voting_authority_none, :accession_number, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (accession_number, cusip, filer_cik, report_date)
            DO UPDATE SET
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append({k: row.get(k) for k in [
                "filer_cik", "filer_name", "report_date", "filing_date", "cusip",
                "issuer_name", "class_title", "market_value", "shares_held",
                "shares_type", "put_call", "investment_discretion",
                "voting_authority_sole", "voting_authority_shared",
                "voting_authority_none", "accession_number",
                "raw_data", "extracted_at", "run_id",
            ]})

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} SEC 13F holdings")
        return rows_loaded

    def load_options_chain(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load options chain data from parquet file."""
        logger.info(f"Loading options data from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_options_chain
            (ticker, quote_date, expiration, strike, option_type, last_price,
             bid, ask, volume, open_interest, implied_volatility, in_the_money,
             raw_data, extracted_at, run_id)
            VALUES (:ticker, :quote_date, :expiration, :strike, :option_type, :last_price,
                    :bid, :ask, :volume, :open_interest, :implied_volatility, :in_the_money,
                    :raw_data, :extracted_at, :run_id)
            ON CONFLICT (ticker, quote_date, expiration, strike, option_type)
            DO UPDATE SET
                last_price = EXCLUDED.last_price,
                implied_volatility = EXCLUDED.implied_volatility,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append({k: row.get(k) for k in [
                "ticker", "quote_date", "expiration", "strike", "option_type",
                "last_price", "bid", "ask", "volume", "open_interest",
                "implied_volatility", "in_the_money",
                "raw_data", "extracted_at", "run_id",
            ]})

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} options contracts")
        return rows_loaded

    def load_earnings_calendar(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load earnings calendar from parquet file."""
        logger.info(f"Loading earnings data from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_earnings_calendar
            (ticker, report_date, fiscal_quarter_end, eps_estimate, eps_actual,
             revenue_estimate, revenue_actual, report_time,
             raw_data, extracted_at, run_id)
            VALUES (:ticker, :report_date, :fiscal_quarter_end, :eps_estimate, :eps_actual,
                    :revenue_estimate, :revenue_actual, :report_time,
                    :raw_data, :extracted_at, :run_id)
            ON CONFLICT (ticker, report_date) DO UPDATE SET
                eps_actual = EXCLUDED.eps_actual,
                revenue_estimate = EXCLUDED.revenue_estimate,
                revenue_actual = EXCLUDED.revenue_actual,
                report_time = EXCLUDED.report_time,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append({k: row.get(k) for k in [
                "ticker", "report_date", "fiscal_quarter_end",
                "eps_estimate", "eps_actual",
                "revenue_estimate", "revenue_actual", "report_time",
                "raw_data", "extracted_at", "run_id",
            ]})

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} earnings records")
        return rows_loaded

    def load_reddit_posts(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load Reddit sentiment posts from parquet file."""
        logger.info(f"Loading Reddit posts from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_reddit_posts
            (post_id, subreddit, title, selftext, author, score,
             upvote_ratio, num_comments, created_utc, tickers_mentioned,
             raw_data, extracted_at, run_id)
            VALUES (:post_id, :subreddit, :title, :selftext, :author, :score,
                    :upvote_ratio, :num_comments, :created_utc, :tickers_mentioned,
                    :raw_data, :extracted_at, :run_id)
            ON CONFLICT (post_id) DO UPDATE SET
                score = EXCLUDED.score,
                num_comments = EXCLUDED.num_comments,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            tickers = row.get("tickers_mentioned")
            records.append({
                "post_id": row.get("post_id"),
                "subreddit": row.get("subreddit"),
                "title": row.get("title"),
                "selftext": row.get("selftext"),
                "author": row.get("author"),
                "score": row.get("score"),
                "upvote_ratio": row.get("upvote_ratio"),
                "num_comments": row.get("num_comments"),
                "created_utc": row.get("created_utc"),
                "tickers_mentioned": json.dumps(tickers) if tickers else None,
                "raw_data": row.get("raw_data"),
                "extracted_at": row.get("extracted_at"),
                "run_id": row.get("run_id"),
            })

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} Reddit posts")
        return rows_loaded

    def load_short_interest(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load short interest data from parquet file."""
        logger.info(f"Loading short interest from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_short_interest
            (ticker, settlement_date, short_interest, avg_daily_volume,
             days_to_cover, raw_data, extracted_at, run_id)
            VALUES (:ticker, :settlement_date, :short_interest, :avg_daily_volume,
                    :days_to_cover, :raw_data, :extracted_at, :run_id)
            ON CONFLICT (ticker, settlement_date) DO UPDATE SET
                short_interest = EXCLUDED.short_interest,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append({k: row.get(k) for k in [
                "ticker", "settlement_date", "short_interest",
                "avg_daily_volume", "days_to_cover",
                "raw_data", "extracted_at", "run_id",
            ]})

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} short interest records")
        return rows_loaded

    def load_etf_flows(self, file_path: Path, run_id: UUID | None = None) -> int:
        """Load ETF flows data from parquet file."""
        logger.info(f"Loading ETF flows from {file_path}")

        df = pd.read_parquet(file_path)
        if df.empty:
            return 0

        insert_sql = text("""
            INSERT INTO raw_etf_flows
            (ticker, date, fund_flow, aum, shares_outstanding,
             raw_data, extracted_at, run_id)
            VALUES (:ticker, :date, :fund_flow, :aum, :shares_outstanding,
                    :raw_data, :extracted_at, :run_id)
            ON CONFLICT (ticker, date) DO UPDATE SET
                aum = EXCLUDED.aum,
                raw_data = EXCLUDED.raw_data,
                extracted_at = EXCLUDED.extracted_at,
                run_id = EXCLUDED.run_id
        """)

        records = []
        for row in df.to_dict(orient="records"):
            row["run_id"] = run_id
            row["raw_data"] = json.dumps(row, default=str)
            row["extracted_at"] = row.get("extracted_at", pd.Timestamp.now())
            records.append({k: row.get(k) for k in [
                "ticker", "date", "fund_flow", "aum", "shares_outstanding",
                "raw_data", "extracted_at", "run_id",
            ]})

        rows_loaded = 0
        with self.db.engine.connect() as conn:
            rows_loaded += self._batch_insert(conn, insert_sql, records)
            conn.commit()

        logger.info(f"Loaded {rows_loaded} ETF flow records")
        return rows_loaded

    def load_all_raw_files(self, raw_dir: Path, source: str, run_id: UUID | None = None) -> int:
        """Load all raw files for a source."""
        source_dir = raw_dir / source
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_dir}")
            return 0

        total_rows = 0
        parquet_files = list(source_dir.rglob("*.parquet"))

        for file_path in parquet_files:
            try:
                if source == "fred":
                    total_rows += self.load_fred_observations(file_path, run_id)
                elif source == "gdelt":
                    total_rows += self.load_gdelt_events(file_path, run_id)
                elif source == "polymarket":
                    if "markets" in str(file_path):
                        total_rows += self.load_polymarket_markets(file_path, run_id)
                    elif "prices" in str(file_path):
                        total_rows += self.load_polymarket_prices(file_path, run_id)
                    elif "trades" in str(file_path):
                        total_rows += self.load_polymarket_trades(file_path, run_id)
                    elif "orderbooks" in str(file_path):
                        total_rows += self.load_polymarket_orderbooks(file_path, run_id)
                elif source == "prices":
                    total_rows += self.load_prices_ohlcv(file_path, run_id)
                elif source == "factors":
                    total_rows += self.load_factor_returns(file_path, run_id)
                elif source == "sec_fundamentals":
                    total_rows += self.load_sec_fundamentals(file_path, run_id)
                elif source == "sec_insider":
                    total_rows += self.load_sec_insider_trades(file_path, run_id)
                elif source == "sec_13f":
                    total_rows += self.load_sec_13f_holdings(file_path, run_id)
                elif source == "options":
                    total_rows += self.load_options_chain(file_path, run_id)
                elif source == "earnings":
                    total_rows += self.load_earnings_calendar(file_path, run_id)
                elif source == "reddit_sentiment":
                    total_rows += self.load_reddit_posts(file_path, run_id)
                elif source == "short_interest":
                    total_rows += self.load_short_interest(file_path, run_id)
                elif source == "etf_flows":
                    total_rows += self.load_etf_flows(file_path, run_id)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue

        return total_rows
