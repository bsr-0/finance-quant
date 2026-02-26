"""Transform raw data into curated tables."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

import pandas as pd
from sqlalchemy import text

from pipeline.db import get_db_manager
from pipeline.historical.latency import get_latency_minutes
from pipeline.infrastructure.lineage import LineageTracker
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)

# A symbol with no trades for this many days is considered delisted.
_DELISTING_GAP_DAYS = 30


class CuratedTransformer:
    """Transform raw data into curated, deduplicated tables."""

    def __init__(self, run_id: str | None = None):
        self.db = get_db_manager()
        self.settings = get_settings()
        self.run_id = run_id
        self._lineage = LineageTracker() if self.settings.infrastructure.lineage_enabled else None

    def _record_lineage(self, source_table: str, target_table: str, transformation_name: str) -> None:
        if not self._lineage:
            return
        try:
            self._lineage.record_lineage(
                source_table=source_table,
                target_table=target_table,
                transformation_name=transformation_name,
                run_id=UUID(self.run_id) if self.run_id else None,
            )
        except Exception as exc:
            logger.warning(f"Lineage recording failed for {transformation_name}: {exc}")

    def _get_source_id(self, source_name: str) -> UUID | None:
        """Get or create source ID."""
        result = self.db.run_query(
            "SELECT source_id FROM dim_source WHERE name = :name",
            {"name": source_name},
        )
        if result:
            return UUID(result[0]["source_id"])

        with self.db.engine.connect() as conn:
            insert = text("""
                INSERT INTO dim_source (name, type, base_url)
                VALUES (:name, :type, :base_url)
                RETURNING source_id
            """)
            result = conn.execute(
                insert,
                {
                    "name": source_name,
                    "type": "api",
                    "base_url": self._get_source_url(source_name),
                },
            )
            source_id = result.scalar()
            conn.commit()
            return source_id

    def _get_source_url(self, source_name: str) -> str:
        """Get base URL for a source."""
        urls = {
            "fred": self.settings.fred.base_url,
            "gdelt": self.settings.gdelt.base_url,
            "polymarket": self.settings.polymarket.base_url,
            "prices": "https://finance.yahoo.com",
        }
        return urls.get(source_name, "")

    def _latency_minutes(self, source_name: str, fallback_minutes: float) -> float:
        fixes = self.settings.historical_fixes
        return get_latency_minutes(
            source_name=source_name,
            percentile=fixes.latency_percentile,
            fallback_minutes=fallback_minutes,
            max_age_hours=fixes.latency_stats_max_age_hours,
            min_samples=fixes.min_latency_samples,
        )

    # ------------------------------------------------------------------
    # P0-3: Macro observations – use FRED realtime_start as available_time
    # ------------------------------------------------------------------

    def transform_macro_observations(self) -> int:
        """Transform raw FRED data to curated macro observations.

        Uses FRED's ``realtime_start`` as the ``available_time`` so that
        point-in-time queries only see data that was genuinely published on
        that date.  Falls back to ``extracted_at`` when ``realtime_start``
        is not available (legacy rows).
        """
        logger.info("Transforming macro observations...")

        source_id = self._get_source_id("fred")

        # Upsert dimension rows for any new series codes
        with self.db.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO dim_macro_series
                    (provider_series_code, name, frequency, source_id,
                     release_time, release_timezone, release_day_offset, release_jitter_minutes)
                SELECT DISTINCT
                    r.series_code,
                    r.series_code AS name,
                    'monthly'     AS frequency,
                    :source_id,
                    :release_time::time,
                    :release_tz,
                    0,
                    :release_jitter
                FROM raw_fred_observations r
                LEFT JOIN dim_macro_series d
                    ON r.series_code = d.provider_series_code
                WHERE d.series_id IS NULL
            """),
                {
                    "source_id": source_id,
                    "release_time": self.settings.historical_fixes.macro_release_time,
                    "release_tz": self.settings.historical_fixes.macro_release_timezone,
                    "release_jitter": self.settings.historical_fixes.macro_release_jitter_minutes,
                },
            )
            conn.commit()

        # Ensure legacy series have conservative release metadata
        with self.db.engine.connect() as conn:
            conn.execute(
                text(
                    """
                    UPDATE dim_macro_series
                    SET release_time = :release_time::time,
                        release_timezone = :release_tz,
                        release_day_offset = COALESCE(release_day_offset, 0),
                        release_jitter_minutes = :release_jitter
                    WHERE release_time IS NULL
                """
                ),
                {
                    "release_time": self.settings.historical_fixes.macro_release_time,
                    "release_tz": self.settings.historical_fixes.macro_release_timezone,
                    "release_jitter": self.settings.historical_fixes.macro_release_jitter_minutes,
                },
            )
            conn.commit()

        # Transform observations.
        # available_time = realtime_start (when FRED first published the value)
        # event_time     = observation_date (period the value refers to)
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_macro_observations
                    (series_id, period_end, value, revision_id,
                     event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    s.series_id,
                    r.observation_date                                AS period_end,
                    r.value,
                    COALESCE(r.realtime_start::text, 'initial')       AS revision_id,
                    r.observation_date::timestamptz                    AS event_time,
                    COALESCE(
                        (r.realtime_start::timestamp + :release_time::time) AT TIME ZONE :release_tz
                            + (:release_jitter || ' minutes')::interval,
                        r.extracted_at
                    )                                                 AS available_time,
                    CASE
                        WHEN r.realtime_start IS NOT NULL THEN 'confirmed'
                        ELSE 'assumed'
                    END                                               AS time_quality
                    ,
                    NOW()                                             AS ingested_at,
                    CASE
                        WHEN r.value IS NULL THEN 'missing_value'
                        ELSE NULL
                    END                                               AS data_quality_flag
                FROM raw_fred_observations r
                JOIN dim_macro_series s
                    ON r.series_code = s.provider_series_code
                ON CONFLICT (series_id, period_end, revision_id) DO UPDATE SET
                    value          = EXCLUDED.value,
                    available_time = EXCLUDED.available_time,
                    time_quality   = EXCLUDED.time_quality,
                    ingested_at    = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag,
                    updated_at     = NOW()
            """),
                {
                    "release_time": self.settings.historical_fixes.macro_release_time,
                    "release_tz": self.settings.historical_fixes.macro_release_timezone,
                    "release_jitter": self.settings.historical_fixes.macro_release_jitter_minutes,
                },
            )
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} macro observations")
        self._record_lineage("raw_fred_observations", "cur_macro_observations", "transform_macro_observations")
        return rows

    def transform_world_events(self) -> int:
        """Transform raw GDELT data to curated world events."""
        logger.info("Transforming world events...")

        source_id = self._get_source_id("gdelt")

        available_source = self.settings.gdelt.available_time_source.upper()
        latency_minutes = self._latency_minutes(
            "gdelt", self.settings.historical_fixes.gdelt_fallback_lag_minutes
        )
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                WITH base AS (
                    SELECT
                        :source_id AS source_id,
                        (r.raw_data::json->>'GLOBALEVENTID')::bigint AS gdelt_event_id,
                        r.raw_data::json->>'EventCode'               AS event_type,
                        COALESCE(
                            (r.raw_data::json->>'SQLDATE')::date::timestamptz,
                            r.extracted_at
                        ) AS event_time,
                        CASE
                            WHEN :available_source = 'DATEADDED' THEN COALESCE(
                                (to_timestamp(NULLIF(r.raw_data::json->>'DATEADDED', ''), 'YYYYMMDDHH24MISS') AT TIME ZONE 'UTC'),
                                r.extracted_at
                            )
                            ELSE r.extracted_at
                        END                                           AS base_available_time,
                        jsonb_build_object(
                            'action_geo_fullname', r.raw_data::json->>'ActionGeo_FullName',
                            'action_geo_country',  r.raw_data::json->>'ActionGeo_CountryCode',
                            'action_geo_lat',      (r.raw_data::json->>'ActionGeo_Lat')::numeric,
                            'action_geo_long',     (r.raw_data::json->>'ActionGeo_Long')::numeric
                        )                                             AS location,
                        jsonb_build_object(
                            'actor1_name', r.raw_data::json->>'Actor1Name',
                            'actor1_code', r.raw_data::json->>'Actor1Code',
                            'actor2_name', r.raw_data::json->>'Actor2Name',
                            'actor2_code', r.raw_data::json->>'Actor2Code'
                        )                                             AS actors,
                        jsonb_build_array(r.raw_data::json->>'EventBaseCode') AS themes,
                        (r.raw_data::json->>'AvgTone')::numeric       AS tone_score,
                        CASE
                            WHEN :available_source = 'DATEADDED' AND r.raw_data::json->>'DATEADDED' IS NOT NULL THEN 'confirmed'
                            ELSE 'assumed'
                        END                                           AS time_quality
                    FROM raw_gdelt_events r
                )
                INSERT INTO cur_world_events
                (source_id, gdelt_event_id, event_type, event_time, available_time,
                 location, actors, themes, tone_score, sentiment_positive,
                 sentiment_negative, time_quality, ingested_at, data_quality_flag)
                SELECT
                    b.source_id,
                    b.gdelt_event_id,
                    b.event_type,
                    b.event_time,
                    GREATEST(
                        b.base_available_time,
                        b.event_time + (:latency_minutes || ' minutes')::interval
                    ) AS available_time,
                    b.location,
                    b.actors,
                    b.themes,
                    b.tone_score,
                    NULL AS sentiment_positive,
                    NULL AS sentiment_negative,
                    b.time_quality,
                    NOW() AS ingested_at,
                    NULL AS data_quality_flag
                FROM base b
                LEFT JOIN cur_world_events c
                    ON b.gdelt_event_id = c.gdelt_event_id
                WHERE c.event_id IS NULL
                ON CONFLICT DO NOTHING
            """),
                {
                    "source_id": source_id,
                    "available_source": available_source,
                    "latency_minutes": latency_minutes,
                },
            )
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} world events")
        self._record_lineage("raw_gdelt_events", "cur_world_events", "transform_world_events")
        return rows

    def transform_contracts(self) -> int:
        """Transform raw Polymarket data to curated contracts."""
        logger.info("Transforming contracts...")

        source_id = self._get_source_id("polymarket")

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO dim_contract
                (venue, venue_market_id, ticker, title, description, category,
                 resolution_time, resolution_rule_text, outcome_type, outcomes, status,
                 created_time, available_time, source_id)
                SELECT
                    'polymarket' AS venue,
                    r.venue_market_id,
                    r.raw_data::json->>'ticker'           AS ticker,
                    r.raw_data::json->>'question'         AS title,
                    r.raw_data::json->>'description'      AS description,
                    r.raw_data::json->>'category'         AS category,
                    (r.raw_data::json->>'resolutionDate')::timestamptz AS resolution_time,
                    r.raw_data::json->>'resolutionSource' AS resolution_rule_text,
                    CASE
                        WHEN (r.raw_data::json->'outcomes')::jsonb @> '["Yes", "No"]'::jsonb
                        THEN 'binary' ELSE 'multi'
                    END AS outcome_type,
                    (r.raw_data::json->'outcomes')::jsonb  AS outcomes,
                    CASE
                        WHEN COALESCE(r.raw_data::json->>'active', 'true')::boolean
                        THEN 'active' ELSE 'closed'
                    END AS status,
                    r.extracted_at                         AS created_time,
                    r.extracted_at                         AS available_time,
                    :source_id
                FROM raw_polymarket_markets r
                LEFT JOIN dim_contract d
                    ON r.venue_market_id = d.venue_market_id AND d.venue = 'polymarket'
                WHERE d.contract_id IS NULL
                ON CONFLICT (venue, venue_market_id) DO UPDATE SET
                    status     = EXCLUDED.status,
                    updated_at = NOW()
            """),
                {"source_id": source_id},
            )
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} contracts")
        self._record_lineage("raw_polymarket_markets", "dim_contract", "transform_contracts")
        return rows

    def transform_contract_state_daily(self) -> int:
        """Populate daily contract state snapshots from contract metadata."""
        logger.info("Transforming contract state daily...")

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                WITH contracts AS (
                    SELECT
                        contract_id,
                        created_time,
                        resolution_time,
                        status,
                        available_time
                    FROM dim_contract
                ),
                dates AS (
                    SELECT
                        c.contract_id,
                        c.status,
                        c.resolution_time,
                        c.available_time,
                        generate_series(
                            date_trunc('day', c.created_time),
                            date_trunc('day', COALESCE(c.resolution_time, NOW())),
                            interval '1 day'
                        ) AS day
                    FROM contracts c
                )
                INSERT INTO cur_contract_state_daily
                    (contract_id, date, status, resolution_time,
                     event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    d.contract_id,
                    d.day::date AS date,
                    CASE
                        WHEN d.resolution_time IS NOT NULL AND d.day::date >= d.resolution_time::date
                        THEN 'resolved'
                        ELSE d.status
                    END AS status,
                    d.resolution_time,
                    d.day::timestamptz AS event_time,
                    GREATEST(d.available_time, d.day::timestamptz) AS available_time,
                    'inferred' AS time_quality,
                    NOW() AS ingested_at,
                    NULL AS data_quality_flag
                FROM dates d
                ON CONFLICT (contract_id, date) DO UPDATE SET
                    status = EXCLUDED.status,
                    resolution_time = EXCLUDED.resolution_time,
                    available_time = EXCLUDED.available_time,
                    time_quality = EXCLUDED.time_quality,
                    ingested_at = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag,
                    updated_at = NOW()
            """))
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} contract state rows")
        self._record_lineage("dim_contract", "cur_contract_state_daily", "transform_contract_state_daily")
        return rows

    def transform_contract_resolution(self) -> int:
        """Populate contract resolution outcomes from market metadata."""
        logger.info("Transforming contract resolutions...")

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_contract_resolution
                    (contract_id, resolved_time, resolved_outcome, resolution_source_url,
                     event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    c.contract_id,
                    COALESCE(
                        (r.raw_data::json->>'resolutionDate')::timestamptz,
                        r.extracted_at
                    ) AS resolved_time,
                    COALESCE(
                        r.raw_data::json->>'resolvedOutcome',
                        r.raw_data::json->>'outcome',
                        r.raw_data::json->>'result'
                    ) AS resolved_outcome,
                    r.raw_data::json->>'resolutionSource' AS resolution_source_url,
                    COALESCE(
                        (r.raw_data::json->>'resolutionDate')::timestamptz,
                        r.extracted_at
                    ) AS event_time,
                    COALESCE(
                        (r.raw_data::json->>'resolutionDate')::timestamptz,
                        r.extracted_at
                    ) AS available_time,
                    CASE
                        WHEN r.raw_data::json->>'resolutionDate' IS NOT NULL THEN 'confirmed'
                        ELSE 'assumed'
                    END AS time_quality,
                    NOW() AS ingested_at,
                    NULL AS data_quality_flag
                FROM raw_polymarket_markets r
                JOIN dim_contract c ON r.venue_market_id = c.venue_market_id
                WHERE COALESCE(
                        r.raw_data::json->>'resolvedOutcome',
                        r.raw_data::json->>'outcome',
                        r.raw_data::json->>'result'
                      ) IS NOT NULL
                ON CONFLICT (contract_id) DO UPDATE SET
                    resolved_time = EXCLUDED.resolved_time,
                    resolved_outcome = EXCLUDED.resolved_outcome,
                    resolution_source_url = EXCLUDED.resolution_source_url,
                    available_time = EXCLUDED.available_time,
                    time_quality = EXCLUDED.time_quality,
                    ingested_at = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag
            """))
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} contract resolutions")
        self._record_lineage("raw_polymarket_markets", "cur_contract_resolution", "transform_contract_resolution")
        return rows

    def transform_contract_prices(self) -> int:
        """Transform raw Polymarket prices to curated contract prices."""
        logger.info("Transforming contract prices...")

        latency_minutes = self._latency_minutes(
            "polymarket", self.settings.historical_fixes.polymarket_fallback_lag_minutes
        )
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_contract_prices
                (contract_id, ts, outcome, price_raw, price_normalized,
                 event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    c.contract_id,
                    r.ts,
                    COALESCE(r.outcome, 'YES') AS outcome,
                    r.price                    AS price_raw,
                    CASE
                        WHEN r.price > 1 THEN r.price / 100.0
                        ELSE r.price
                    END                        AS price_normalized,
                    r.ts                       AS event_time,
                    r.ts + (:latency_minutes || ' minutes')::interval AS available_time,
                    'inferred'                 AS time_quality,
                    NOW()                      AS ingested_at,
                    CASE
                        WHEN r.price IS NULL OR r.price <= 0 THEN 'invalid_price'
                        WHEN (CASE WHEN r.price > 1 THEN r.price / 100.0 ELSE r.price END) > 1 THEN 'price_out_of_range'
                        WHEN (CASE WHEN r.price > 1 THEN r.price / 100.0 ELSE r.price END) < 0 THEN 'price_out_of_range'
                        ELSE NULL
                    END                        AS data_quality_flag
                FROM raw_polymarket_prices r
                JOIN dim_contract c ON r.venue_market_id = c.venue_market_id
                ON CONFLICT (contract_id, ts, outcome) DO UPDATE SET
                    price_raw        = EXCLUDED.price_raw,
                    price_normalized = EXCLUDED.price_normalized,
                    available_time   = EXCLUDED.available_time,
                    ingested_at      = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag
            """), {"latency_minutes": latency_minutes})
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} contract prices")
        self._record_lineage("raw_polymarket_prices", "cur_contract_prices", "transform_contract_prices")
        return rows

    def transform_contract_trades(self) -> int:
        """Transform raw Polymarket trades to curated contract trades."""
        logger.info("Transforming contract trades...")

        latency_minutes = self._latency_minutes(
            "polymarket", self.settings.historical_fixes.polymarket_fallback_lag_minutes
        )
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_contract_trades
                (contract_id, trade_id, ts, price, size, side,
                 event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    c.contract_id,
                    r.trade_id,
                    r.ts,
                    r.price,
                    r.size,
                    r.side,
                    r.ts           AS event_time,
                    r.ts + (:latency_minutes || ' minutes')::interval AS available_time,
                    'inferred'     AS time_quality,
                    NOW()          AS ingested_at,
                    CASE
                        WHEN r.price IS NULL OR r.price <= 0 THEN 'invalid_price'
                        WHEN r.size IS NULL OR r.size <= 0 THEN 'invalid_size'
                        ELSE NULL
                    END            AS data_quality_flag
                FROM raw_polymarket_trades r
                JOIN dim_contract c ON r.venue_market_id = c.venue_market_id
                ON CONFLICT (contract_id, trade_id) DO NOTHING
            """),
                {"latency_minutes": latency_minutes},
            )
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} contract trades")
        self._record_lineage("raw_polymarket_trades", "cur_contract_trades", "transform_contract_trades")
        return rows

    def transform_contract_orderbooks(self) -> int:
        """Transform raw Polymarket orderbook snapshots to curated snapshots."""
        logger.info("Transforming contract orderbook snapshots...")

        if not self.db.table_exists("raw_polymarket_orderbook_snapshots"):
            logger.info("raw_polymarket_orderbook_snapshots does not exist; skipping")
            return 0

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_contract_orderbook_snapshots
                (contract_id, ts, best_bid, best_ask, spread, bids, asks,
                 event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    c.contract_id,
                    r.ts,
                    r.best_bid,
                    r.best_ask,
                    r.spread,
                    r.bids,
                    r.asks,
                    r.ts          AS event_time,
                    r.ts          AS available_time,
                    'inferred'    AS time_quality,
                    NOW()         AS ingested_at,
                    NULL          AS data_quality_flag
                FROM raw_polymarket_orderbook_snapshots r
                JOIN dim_contract c ON r.venue_market_id = c.venue_market_id
                ON CONFLICT (contract_id, ts) DO UPDATE SET
                    best_bid = EXCLUDED.best_bid,
                    best_ask = EXCLUDED.best_ask,
                    spread = EXCLUDED.spread,
                    bids = EXCLUDED.bids,
                    asks = EXCLUDED.asks,
                    available_time = EXCLUDED.available_time,
                    ingested_at = EXCLUDED.ingested_at
            """))
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} orderbook snapshots")
        self._record_lineage(
            "raw_polymarket_orderbook_snapshots",
            "cur_contract_orderbook_snapshots",
            "transform_contract_orderbooks",
        )
        return rows

    def transform_factor_returns(self) -> int:
        """Transform raw factor returns to curated factor table."""
        logger.info("Transforming factor returns...")

        if not self.db.table_exists("raw_factor_returns"):
            logger.info("raw_factor_returns does not exist; skipping")
            return 0

        lag_days = self.settings.factors.available_time_lag_days
        release_time = self.settings.factors.available_time_release_time
        tz = self.settings.factors.exchange_timezone
        jitter_minutes = self.settings.historical_fixes.macro_release_jitter_minutes

        with self.db.engine.connect() as conn:
            df = pd.read_sql("SELECT date, mkt_rf, smb, hml, rmw, cma, mom, rf FROM raw_factor_returns", conn)
            if df.empty:
                return 0

            dates = pd.to_datetime(df["date"])
            available_dates = dates + pd.tseries.offsets.BDay(lag_days)
            available_time = pd.to_datetime(
                available_dates.dt.date.astype(str) + " " + release_time
            ).dt.tz_localize(tz).dt.tz_convert("UTC")
            if jitter_minutes:
                available_time = available_time + pd.to_timedelta(jitter_minutes, unit="m")

            df["event_time"] = dates.dt.tz_localize("UTC")
            df["available_time"] = available_time
            df["time_quality"] = "inferred"
            df["ingested_at"] = datetime.now(timezone.utc)
            df["data_quality_flag"] = None

            insert_sql = text("""
                INSERT INTO cur_factor_returns
                (date, mkt_rf, smb, hml, rmw, cma, mom, rf,
                 event_time, available_time, time_quality, ingested_at, data_quality_flag)
                VALUES
                (:date, :mkt_rf, :smb, :hml, :rmw, :cma, :mom, :rf,
                 :event_time, :available_time, :time_quality, :ingested_at, :data_quality_flag)
                ON CONFLICT (date) DO UPDATE SET
                    mkt_rf = EXCLUDED.mkt_rf,
                    smb = EXCLUDED.smb,
                    hml = EXCLUDED.hml,
                    rmw = EXCLUDED.rmw,
                    cma = EXCLUDED.cma,
                    mom = EXCLUDED.mom,
                    rf = EXCLUDED.rf,
                    available_time = EXCLUDED.available_time,
                    time_quality = EXCLUDED.time_quality,
                    ingested_at = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag,
                    updated_at = NOW()
            """)
            records = df.to_dict(orient="records")
            conn.execute(insert_sql, records)
            conn.commit()
            rows = len(records)

        logger.info(f"Transformed {rows} factor rows")
        self._record_lineage("raw_factor_returns", "cur_factor_returns", "transform_factor_returns")
        return rows

    # ------------------------------------------------------------------
    # P0-1: Survivor bias – detect delisted symbols + mark dim_symbol
    # P0-2: Corporate actions – populate cur_corporate_actions
    # ------------------------------------------------------------------

    def transform_prices_ohlcv(self) -> int:
        """Transform raw OHLCV data to curated prices.

        Also populates ``cur_corporate_actions`` from split / dividend data
        captured by the extractor, and detects delisted symbols.
        """
        logger.info("Transforming OHLCV prices...")
        close_time = self.settings.prices.market_close_time
        exchange_tz = self.settings.prices.exchange_timezone
        base_delay = self.settings.prices.vendor_delay_minutes
        latency_delay = self._latency_minutes("prices", base_delay)
        delay_minutes = max(base_delay, latency_delay)

        # ------ 1. Upsert dim_symbol (new tickers only) ------
        with self.db.engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO dim_symbol
                    (ticker, exchange, asset_class, currency, start_date, is_delisted)
                SELECT DISTINCT
                    r.ticker,
                    'NYSE'   AS exchange,
                    'equity' AS asset_class,
                    'USD'    AS currency,
                    MIN(r.date) AS start_date,
                    false    AS is_delisted
                FROM raw_prices_ohlcv r
                LEFT JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE s.symbol_id IS NULL
                GROUP BY r.ticker
            """)
            )
            conn.commit()

        # ------ 2. Detect delistings ------
        # If the latest trading day for a ticker is >30 days before the
        # global max trading day across all tickers, mark it delisted.
        with self.db.engine.connect() as conn:
            conn.execute(text("""
                WITH last_dates AS (
                    SELECT ticker, MAX(date) AS last_date
                    FROM raw_prices_ohlcv
                    GROUP BY ticker
                ),
                global_max AS (
                    SELECT MAX(last_date) AS max_date FROM last_dates
                )
                UPDATE dim_symbol s
                SET is_delisted = true,
                    end_date    = ld.last_date
                FROM last_dates ld, global_max gm
                WHERE s.ticker = ld.ticker
                  AND ld.last_date < gm.max_date - INTERVAL ':gap days'
            """).bindparams(gap=_DELISTING_GAP_DAYS))
            conn.commit()
            logger.info("Updated delisting flags on dim_symbol")

        # ------ 3. Populate cur_corporate_actions (splits) ------
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_corporate_actions
                    (symbol_id, action_type, action_date, ratio,
                     event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    s.symbol_id,
                    'split'            AS action_type,
                    r.date             AS action_date,
                    -- split_ratio is stored as e.g. '4:1'; extract the
                    -- numeric multiplier (shares-after / shares-before).
                    CASE
                        WHEN r.split_ratio LIKE '%:%' THEN
                            SPLIT_PART(r.split_ratio, ':', 1)::numeric
                            / NULLIF(SPLIT_PART(r.split_ratio, ':', 2)::numeric, 0)
                        ELSE NULL
                    END                AS ratio,
                    (r.date + :close_time::time) AT TIME ZONE :exchange_tz AS event_time,
                    (r.date + :close_time::time) AT TIME ZONE :exchange_tz
                        + (:delay_minutes || ' minutes')::interval AS available_time,
                    'inferred'         AS time_quality,
                    NOW()              AS ingested_at,
                    NULL               AS data_quality_flag
                FROM raw_prices_ohlcv r
                JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE r.split_ratio IS NOT NULL
                ON CONFLICT (symbol_id, action_type, action_date) DO NOTHING
            """), {"close_time": close_time, "exchange_tz": exchange_tz, "delay_minutes": delay_minutes})
            conn.commit()
            splits = result.rowcount
            if splits:
                logger.info(f"Inserted {splits} split actions")

        # ------ 4. Populate cur_corporate_actions (dividends) ------
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_corporate_actions
                    (symbol_id, action_type, action_date, amount,
                     event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    s.symbol_id,
                    'dividend'         AS action_type,
                    r.date             AS action_date,
                    r.dividend         AS amount,
                    (r.date + :close_time::time) AT TIME ZONE :exchange_tz AS event_time,
                    (r.date + :close_time::time) AT TIME ZONE :exchange_tz
                        + (:delay_minutes || ' minutes')::interval AS available_time,
                    'inferred'         AS time_quality,
                    NOW()              AS ingested_at,
                    NULL               AS data_quality_flag
                FROM raw_prices_ohlcv r
                JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE r.dividend IS NOT NULL AND r.dividend > 0
                ON CONFLICT (symbol_id, action_type, action_date) DO NOTHING
            """), {"close_time": close_time, "exchange_tz": exchange_tz, "delay_minutes": delay_minutes})
            conn.commit()
            divs = result.rowcount
            if divs:
                logger.info(f"Inserted {divs} dividend actions")

        # ------ 5. Transform prices ------
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_prices_ohlcv_daily
                    (symbol_id, date, open, high, low, close, adj_close, volume,
                     event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    s.symbol_id,
                    r.date,
                    r.open,
                    r.high,
                    r.low,
                    r.close,
                    r.adj_close,
                    r.volume,
                    (r.date + :close_time::time) AT TIME ZONE :exchange_tz AS event_time,
                    (r.date + :close_time::time) AT TIME ZONE :exchange_tz
                        + (:delay_minutes || ' minutes')::interval AS available_time,
                    'inferred'                             AS time_quality,
                    NOW()                                  AS ingested_at,
                    CASE
                        WHEN r.open < 0 OR r.high < 0 OR r.low < 0 OR r.close < 0 THEN 'negative_price'
                        WHEN r.low > r.high OR r.open > r.high OR r.open < r.low OR r.close > r.high OR r.close < r.low THEN 'ohlc_error'
                        WHEN r.volume < 0 THEN 'negative_volume'
                        ELSE NULL
                    END                                    AS data_quality_flag
                FROM raw_prices_ohlcv r
                JOIN dim_symbol s ON r.ticker = s.ticker
                ON CONFLICT (symbol_id, date) DO UPDATE SET
                    open           = EXCLUDED.open,
                    high           = EXCLUDED.high,
                    low            = EXCLUDED.low,
                    close          = EXCLUDED.close,
                    adj_close      = EXCLUDED.adj_close,
                    volume         = EXCLUDED.volume,
                    available_time = EXCLUDED.available_time,
                    time_quality   = EXCLUDED.time_quality,
                    ingested_at    = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag,
                    updated_at     = NOW()
            """), {"close_time": close_time, "exchange_tz": exchange_tz, "delay_minutes": delay_minutes})
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} OHLCV records")

        # ------ 6. Flag data quality issues ------
        # Priority: invalid_price > ohlc_error > zero_volume (price_spike handled separately)
        with self.db.engine.connect() as conn:
            conn.execute(text("""
                UPDATE cur_prices_ohlcv_daily
                SET data_quality_flag = CASE
                    WHEN close <= 0 OR close IS NULL          THEN 'invalid_price'
                    WHEN high < low                            THEN 'ohlc_error'
                    WHEN volume = 0 OR volume IS NULL          THEN 'zero_volume'
                    ELSE 'ok'
                END
                WHERE data_quality_flag IS NULL OR data_quality_flag = 'ok'
            """))
            conn.commit()

        # Flag price spikes (>50% daily change) – requires LAG, separate pass
        with self.db.engine.connect() as conn:
            conn.execute(text("""
                WITH price_changes AS (
                    SELECT
                        symbol_id,
                        date,
                        ABS(close / NULLIF(
                            LAG(close) OVER (PARTITION BY symbol_id ORDER BY date), 0
                        ) - 1) AS pct_change
                    FROM cur_prices_ohlcv_daily
                )
                UPDATE cur_prices_ohlcv_daily p
                SET data_quality_flag = 'price_spike'
                FROM price_changes pc
                WHERE p.symbol_id = pc.symbol_id
                  AND p.date = pc.date
                  AND pc.pct_change IS NOT NULL
                  AND pc.pct_change > 0.50
                  AND p.data_quality_flag = 'ok'
            """))
            conn.commit()
            logger.info("Applied data quality flags to OHLCV records")

        self._record_lineage("raw_prices_ohlcv", "cur_prices_ohlcv_daily", "transform_prices_ohlcv")
        return rows

    def transform_prices_adjusted_daily(self) -> int:
        """Build point-in-time total-return adjusted prices (splits + dividends)."""
        logger.info("Transforming adjusted OHLCV prices...")

        if not self.db.table_exists("cur_prices_ohlcv_daily"):
            logger.info("cur_prices_ohlcv_daily does not exist; skipping")
            return 0

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                WITH price_base AS (
                    SELECT
                        p.symbol_id,
                        p.date,
                        p.open,
                        p.high,
                        p.low,
                        p.close,
                        p.volume,
                        p.event_time,
                        p.available_time,
                        p.time_quality,
                        COALESCE(a.split_ratio, 1) AS split_ratio,
                        COALESCE(a.dividend, 0) AS dividend
                    FROM cur_prices_ohlcv_daily p
                    LEFT JOIN LATERAL (
                        SELECT
                            MAX(CASE WHEN action_type = 'split' THEN ratio END) AS split_ratio,
                            SUM(CASE WHEN action_type = 'dividend' THEN amount END) AS dividend
                        FROM cur_corporate_actions ca
                        WHERE ca.symbol_id = p.symbol_id
                          AND ca.action_date = p.date
                          AND ca.available_time <= p.available_time
                    ) a ON TRUE
                ),
                factors AS (
                    SELECT
                        pb.*,
                        LAG(pb.close) OVER (PARTITION BY pb.symbol_id ORDER BY pb.date) AS prev_close
                    FROM price_base pb
                ),
                returns AS (
                    SELECT
                        f.*,
                        CASE
                            WHEN f.prev_close IS NULL OR f.prev_close = 0 THEN 1
                            ELSE (f.close * f.split_ratio + f.dividend) / f.prev_close
                        END AS total_return_factor,
                        CASE
                            WHEN f.split_ratio IS NULL OR f.split_ratio = 0 THEN 1
                            ELSE f.split_ratio
                        END AS split_ratio_safe
                    FROM factors f
                ),
                cum AS (
                    SELECT
                        r.*,
                        EXP(
                            SUM(
                                LN(
                                    CASE WHEN r.total_return_factor <= 0 THEN 1 ELSE r.total_return_factor END
                                )
                            ) OVER (PARTITION BY r.symbol_id ORDER BY r.date)
                        ) AS total_return_index,
                        EXP(
                            SUM(LN(r.split_ratio_safe)) OVER (PARTITION BY r.symbol_id ORDER BY r.date)
                        ) AS split_cum_factor,
                        FIRST_VALUE(r.close) OVER (PARTITION BY r.symbol_id ORDER BY r.date) AS base_close
                    FROM returns r
                )
                INSERT INTO cur_prices_adjusted_daily
                    (symbol_id, date, adj_open, adj_high, adj_low, adj_close, adj_volume,
                     adj_factor, event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    symbol_id,
                    date,
                    CASE WHEN close = 0 THEN NULL ELSE open * (base_close * total_return_index / close) END AS adj_open,
                    CASE WHEN close = 0 THEN NULL ELSE high * (base_close * total_return_index / close) END AS adj_high,
                    CASE WHEN close = 0 THEN NULL ELSE low * (base_close * total_return_index / close) END AS adj_low,
                    CASE WHEN close = 0 THEN NULL ELSE (base_close * total_return_index) END AS adj_close,
                    volume * split_cum_factor AS adj_volume,
                    CASE WHEN close = 0 THEN NULL ELSE (base_close * total_return_index / close) END AS adj_factor,
                    event_time,
                    available_time,
                    time_quality,
                    NOW() AS ingested_at,
                    NULL  AS data_quality_flag
                FROM cum
                ON CONFLICT (symbol_id, date) DO UPDATE SET
                    adj_open = EXCLUDED.adj_open,
                    adj_high = EXCLUDED.adj_high,
                    adj_low = EXCLUDED.adj_low,
                    adj_close = EXCLUDED.adj_close,
                    adj_volume = EXCLUDED.adj_volume,
                    adj_factor = EXCLUDED.adj_factor,
                    available_time = EXCLUDED.available_time,
                    time_quality = EXCLUDED.time_quality,
                    ingested_at = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag,
                    updated_at = NOW()
            """))
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} adjusted OHLCV records")
        self._record_lineage(
            "cur_prices_ohlcv_daily",
            "cur_prices_adjusted_daily",
            "transform_prices_adjusted_daily",
        )
        return rows

    def transform_universe_membership(self) -> int:
        """Build point-in-time universe membership snapshots."""
        logger.info("Transforming universe membership snapshots...")

        close_time = self.settings.prices.market_close_time
        exchange_tz = self.settings.prices.exchange_timezone
        base_delay = self.settings.prices.vendor_delay_minutes
        latency_delay = self._latency_minutes("prices", base_delay)
        delay_minutes = max(base_delay, latency_delay)

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO snap_universe_membership
                    (symbol_id, start_date, end_date, is_delisted, available_time, ingested_at)
                SELECT
                    s.symbol_id,
                    s.start_date,
                    s.end_date,
                    s.is_delisted,
                    (s.start_date + :close_time::time) AT TIME ZONE :exchange_tz
                        + (:delay_minutes || ' minutes')::interval AS available_time,
                    NOW() AS ingested_at
                FROM dim_symbol s
                WHERE s.start_date IS NOT NULL
                ON CONFLICT (symbol_id, start_date) DO UPDATE SET
                    end_date = EXCLUDED.end_date,
                    is_delisted = EXCLUDED.is_delisted,
                    available_time = EXCLUDED.available_time,
                    ingested_at = EXCLUDED.ingested_at
            """), {"close_time": close_time, "exchange_tz": exchange_tz, "delay_minutes": delay_minutes})
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} universe membership records")
        self._record_lineage("dim_symbol", "snap_universe_membership", "transform_universe_membership")
        return rows

    # ------------------------------------------------------------------

    def transform_all(self) -> dict:
        """Run all transformations."""
        results = {}
        results["macro_observations"] = self.transform_macro_observations()
        results["world_events"] = self.transform_world_events()
        results["contracts"] = self.transform_contracts()
        results["contract_state_daily"] = self.transform_contract_state_daily()
        results["contract_prices"] = self.transform_contract_prices()
        results["contract_trades"] = self.transform_contract_trades()
        results["contract_orderbooks"] = self.transform_contract_orderbooks()
        results["contract_resolution"] = self.transform_contract_resolution()
        results["prices_ohlcv"] = self.transform_prices_ohlcv()
        results["prices_adjusted"] = self.transform_prices_adjusted_daily()
        results["universe_membership"] = self.transform_universe_membership()
        results["factor_returns"] = self.transform_factor_returns()
        return results
