"""Transform raw data into curated tables."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
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

    @staticmethod
    def _resolve_rowcount(result, conn, target_table: str) -> int:
        """Return the number of affected rows, querying the table if the driver returns -1."""
        rows = result.rowcount
        if rows >= 0:
            return rows
        # DuckDB returns -1 for INSERT...SELECT; fall back to a COUNT query.
        from pipeline.db import _validate_identifier

        count_result = conn.execute(
            text(f"SELECT COUNT(*) AS n FROM {_validate_identifier(target_table)}")
        )
        return count_result.scalar() or 0

    def _record_lineage(
        self, source_table: str, target_table: str, transformation_name: str
    ) -> None:
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
            val = result[0]["source_id"]
            return val if isinstance(val, UUID) else UUID(val)

        with self.db.engine.connect() as conn:
            insert = text("""
                INSERT INTO dim_source (name, type, base_url)
                VALUES (:name, :type, :base_url)
                RETURNING source_id
            """)
            cursor_result = conn.execute(
                insert,
                {
                    "name": source_name,
                    "type": "api",
                    "base_url": self._get_source_url(source_name),
                },
            )
            source_id = cursor_result.scalar()
            conn.commit()
            return source_id

    def _get_source_url(self, source_name: str) -> str:
        """Get base URL for a source."""
        urls = {
            "fred": self.settings.fred.base_url,
            "gdelt": self.settings.gdelt.base_url,
            "polymarket": self.settings.polymarket.base_url,
            "prices": "https://finance.yahoo.com",
            "sec_edgar": "https://data.sec.gov",
            "options": "https://finance.yahoo.com",
            "earnings": "https://finance.yahoo.com",
            "reddit": "https://www.reddit.com",
            "short_interest": "https://www.finra.org",
            "etf_flows": "https://finance.yahoo.com",
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
            conn.execute(
                text("""
                INSERT INTO dim_macro_series
                    (provider_series_code, name, frequency, source_id,
                     release_time, release_timezone, release_day_offset, release_jitter_minutes)
                SELECT DISTINCT
                    r.series_code,
                    r.series_code AS name,
                    'monthly'     AS frequency,
                    :source_id,
                    CAST(:release_time AS TIME),
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
                text("""
                    UPDATE dim_macro_series
                    SET release_time = CAST(:release_time AS TIME),
                        release_timezone = :release_tz,
                        release_day_offset = COALESCE(release_day_offset, 0),
                        release_jitter_minutes = :release_jitter
                    WHERE release_time IS NULL
                """),
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
            result = conn.execute(
                text("""
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
                        (r.realtime_start::date + CAST(:release_time AS TIME))
                            AT TIME ZONE :release_tz
                            + (CAST(:release_jitter AS INTEGER) * INTERVAL '1' MINUTE),
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
            rows = self._resolve_rowcount(result, conn, "cur_macro_observations")

        logger.info(f"Transformed {rows} macro observations")
        self._record_lineage(
            "raw_fred_observations", "cur_macro_observations", "transform_macro_observations"
        )
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
            result = conn.execute(
                text("""
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
                                (strptime(
                                    RPAD(NULLIF(r.raw_data::json->>'DATEADDED', ''), 14, '0'),
                                    '%Y%m%d%H%M%S'
                                ) AT TIME ZONE 'UTC'),
                                r.extracted_at
                            )
                            ELSE r.extracted_at
                        END                                           AS base_available_time,
                        json_object(
                            'action_geo_fullname', r.raw_data::json->>'ActionGeo_FullName',
                            'action_geo_country',  r.raw_data::json->>'ActionGeo_CountryCode',
                            'action_geo_lat',      (r.raw_data::json->>'ActionGeo_Lat')::numeric,
                            'action_geo_long',     (r.raw_data::json->>'ActionGeo_Long')::numeric
                        )                                             AS location,
                        json_object(
                            'actor1_name', r.raw_data::json->>'Actor1Name',
                            'actor1_code', r.raw_data::json->>'Actor1Code',
                            'actor2_name', r.raw_data::json->>'Actor2Name',
                            'actor2_code', r.raw_data::json->>'Actor2Code'
                        )                                             AS actors,
                        json_array(r.raw_data::json->>'EventBaseCode') AS themes,
                        (r.raw_data::json->>'AvgTone')::numeric       AS tone_score,
                        CASE
                            WHEN :available_source = 'DATEADDED'
                                AND r.raw_data::json->>'DATEADDED' IS NOT NULL
                                THEN 'confirmed'
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
                        b.event_time + (CAST(:latency_minutes AS INTEGER) * INTERVAL '1' MINUTE)
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
            rows = self._resolve_rowcount(result, conn, "cur_world_events")

        logger.info(f"Transformed {rows} world events")
        self._record_lineage("raw_gdelt_events", "cur_world_events", "transform_world_events")
        return rows

    def transform_contracts(self) -> int:
        """Transform raw Polymarket data to curated contracts."""
        logger.info("Transforming contracts...")

        source_id = self._get_source_id("polymarket")

        with self.db.engine.connect() as conn:
            result = conn.execute(
                text("""
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
                        WHEN list_contains(CAST(r.raw_data::json->'outcomes' AS VARCHAR[]), 'Yes')
                         AND list_contains(CAST(r.raw_data::json->'outcomes' AS VARCHAR[]), 'No')
                        THEN 'binary' ELSE 'multi'
                    END AS outcome_type,
                    CAST(r.raw_data::json->'outcomes' AS JSON)  AS outcomes,
                    CASE
                        WHEN CAST(COALESCE(r.raw_data::json->>'active', 'true') AS BOOLEAN)
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
            rows = self._resolve_rowcount(result, conn, "dim_contract")

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
                        unnest(generate_series(
                            date_trunc('day', c.created_time),
                            date_trunc('day', COALESCE(c.resolution_time, NOW())),
                            interval '1 day'
                        )) AS day
                    FROM contracts c
                )
                INSERT INTO cur_contract_state_daily
                    (contract_id, date, status, resolution_time,
                     event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    d.contract_id,
                    d.day::date AS date,
                    CASE
                        WHEN d.resolution_time IS NOT NULL
                            AND d.day::date >= d.resolution_time::date
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
            rows = self._resolve_rowcount(result, conn, "cur_contract_state_daily")

        logger.info(f"Transformed {rows} contract state rows")
        self._record_lineage(
            "dim_contract", "cur_contract_state_daily", "transform_contract_state_daily"
        )
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
            rows = self._resolve_rowcount(result, conn, "cur_contract_resolution")

        logger.info(f"Transformed {rows} contract resolutions")
        self._record_lineage(
            "raw_polymarket_markets", "cur_contract_resolution", "transform_contract_resolution"
        )
        return rows

    def transform_contract_prices(self) -> int:
        """Transform raw Polymarket prices to curated contract prices."""
        logger.info("Transforming contract prices...")

        latency_minutes = self._latency_minutes(
            "polymarket", self.settings.historical_fixes.polymarket_fallback_lag_minutes
        )
        with self.db.engine.connect() as conn:
            result = conn.execute(
                text("""
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
                    r.ts + (CAST(:latency_minutes AS INTEGER)
                        * INTERVAL '1' MINUTE) AS available_time,
                    'inferred'                 AS time_quality,
                    NOW()                      AS ingested_at,
                    CASE
                        WHEN r.price IS NULL OR r.price <= 0
                            THEN 'invalid_price'
                        WHEN (CASE WHEN r.price > 1
                            THEN r.price / 100.0 ELSE r.price END) > 1
                            THEN 'price_out_of_range'
                        WHEN (CASE WHEN r.price > 1
                            THEN r.price / 100.0 ELSE r.price END) < 0
                            THEN 'price_out_of_range'
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
            """),
                {"latency_minutes": latency_minutes},
            )
            conn.commit()
            rows = self._resolve_rowcount(result, conn, "cur_contract_prices")

        logger.info(f"Transformed {rows} contract prices")
        self._record_lineage(
            "raw_polymarket_prices", "cur_contract_prices", "transform_contract_prices"
        )
        return rows

    def transform_contract_trades(self) -> int:
        """Transform raw Polymarket trades to curated contract trades."""
        logger.info("Transforming contract trades...")

        latency_minutes = self._latency_minutes(
            "polymarket", self.settings.historical_fixes.polymarket_fallback_lag_minutes
        )
        with self.db.engine.connect() as conn:
            result = conn.execute(
                text("""
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
                    r.ts + (CAST(:latency_minutes AS INTEGER)
                        * INTERVAL '1' MINUTE) AS available_time,
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
            rows = self._resolve_rowcount(result, conn, "cur_contract_trades")

        logger.info(f"Transformed {rows} contract trades")
        self._record_lineage(
            "raw_polymarket_trades", "cur_contract_trades", "transform_contract_trades"
        )
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
            rows = self._resolve_rowcount(result, conn, "cur_contract_orderbook_snapshots")

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
            df = pd.read_sql(
                text("SELECT date, mkt_rf, smb, hml, rmw, cma, mom, rf FROM raw_factor_returns"), conn
            )
            if df.empty:
                return 0

            dates = pd.to_datetime(df["date"])
            available_dates = dates + pd.tseries.offsets.BDay(lag_days)
            available_time = (
                pd.to_datetime(available_dates.dt.date.astype(str) + " " + release_time)
                .dt.tz_localize(tz)
                .dt.tz_convert("UTC")
            )
            if jitter_minutes:
                available_time = available_time + pd.to_timedelta(jitter_minutes, unit="m")

            df["event_time"] = dates.dt.tz_localize("UTC")
            df["available_time"] = available_time
            df["time_quality"] = "inferred"
            df["ingested_at"] = datetime.now(UTC)
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
            """))
            conn.commit()

        # ------ 2. Detect delistings ------
        # If the latest trading day for a ticker is >30 days before the
        # global max trading day across all tickers, mark it delisted.
        with self.db.engine.connect() as conn:
            conn.execute(
                text("""
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
                  AND ld.last_date < gm.max_date - (CAST(:gap AS INTEGER) * INTERVAL '1' DAY)
            """),
                {"gap": _DELISTING_GAP_DAYS},
            )
            conn.commit()
            logger.info("Updated delisting flags on dim_symbol")

        # ------ 3. Populate cur_corporate_actions (splits) ------
        with self.db.engine.connect() as conn:
            result = conn.execute(
                text("""
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
                    (r.date + CAST(:close_time AS TIME)) AT TIME ZONE :exchange_tz AS event_time,
                    (r.date + CAST(:close_time AS TIME)) AT TIME ZONE :exchange_tz
                        + (CAST(:delay_minutes AS INTEGER) * INTERVAL '1' MINUTE) AS available_time,
                    'inferred'         AS time_quality,
                    NOW()              AS ingested_at,
                    NULL               AS data_quality_flag
                FROM raw_prices_ohlcv r
                JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE r.split_ratio IS NOT NULL
                ON CONFLICT (symbol_id, action_type, action_date) DO NOTHING
            """),
                {
                    "close_time": close_time,
                    "exchange_tz": exchange_tz,
                    "delay_minutes": delay_minutes,
                },
            )
            conn.commit()
            splits = self._resolve_rowcount(result, conn, "cur_corporate_actions")
            if splits:
                logger.info(f"Inserted {splits} split actions")

        # ------ 4. Populate cur_corporate_actions (dividends) ------
        with self.db.engine.connect() as conn:
            result = conn.execute(
                text("""
                INSERT INTO cur_corporate_actions
                    (symbol_id, action_type, action_date, amount,
                     event_time, available_time, time_quality, ingested_at, data_quality_flag)
                SELECT
                    s.symbol_id,
                    'dividend'         AS action_type,
                    r.date             AS action_date,
                    r.dividend         AS amount,
                    (r.date + CAST(:close_time AS TIME)) AT TIME ZONE :exchange_tz AS event_time,
                    (r.date + CAST(:close_time AS TIME)) AT TIME ZONE :exchange_tz
                        + (CAST(:delay_minutes AS INTEGER) * INTERVAL '1' MINUTE) AS available_time,
                    'inferred'         AS time_quality,
                    NOW()              AS ingested_at,
                    NULL               AS data_quality_flag
                FROM raw_prices_ohlcv r
                JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE r.dividend IS NOT NULL AND r.dividend > 0
                ON CONFLICT (symbol_id, action_type, action_date) DO NOTHING
            """),
                {
                    "close_time": close_time,
                    "exchange_tz": exchange_tz,
                    "delay_minutes": delay_minutes,
                },
            )
            conn.commit()
            divs = self._resolve_rowcount(result, conn, "cur_corporate_actions")
            if divs:
                logger.info(f"Inserted {divs} dividend actions")

        # ------ 5. Transform prices ------
        with self.db.engine.connect() as conn:
            result = conn.execute(
                text("""
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
                    (r.date + CAST(:close_time AS TIME)) AT TIME ZONE :exchange_tz AS event_time,
                    (r.date + CAST(:close_time AS TIME)) AT TIME ZONE :exchange_tz
                        + (CAST(:delay_minutes AS INTEGER) * INTERVAL '1' MINUTE) AS available_time,
                    'inferred'                             AS time_quality,
                    NOW()                                  AS ingested_at,
                    CASE
                        WHEN r.open < 0 OR r.high < 0
                            OR r.low < 0 OR r.close < 0
                            THEN 'negative_price'
                        WHEN r.low > r.high OR r.open > r.high
                            OR r.open < r.low OR r.close > r.high
                            OR r.close < r.low
                            THEN 'ohlc_error'
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
            """),
                {
                    "close_time": close_time,
                    "exchange_tz": exchange_tz,
                    "delay_minutes": delay_minutes,
                },
            )
            conn.commit()
            rows = self._resolve_rowcount(result, conn, "cur_prices_ohlcv_daily")

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
                        LAG(pb.close) OVER (
                            PARTITION BY pb.symbol_id ORDER BY pb.date
                        ) AS prev_close
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
                                    CASE WHEN r.total_return_factor <= 0
                                        THEN 1
                                        ELSE r.total_return_factor
                                    END
                                )
                            ) OVER (PARTITION BY r.symbol_id ORDER BY r.date)
                        ) AS total_return_index,
                        EXP(
                            SUM(LN(r.split_ratio_safe)) OVER (
                                PARTITION BY r.symbol_id ORDER BY r.date
                            )
                        ) AS split_cum_factor,
                        FIRST_VALUE(r.close) OVER (
                            PARTITION BY r.symbol_id ORDER BY r.date
                        ) AS base_close
                    FROM returns r
                )
                INSERT INTO cur_prices_adjusted_daily
                    (symbol_id, date, adj_open, adj_high, adj_low, adj_close, adj_volume,
                     adj_factor, event_time, available_time,
                     time_quality, ingested_at, data_quality_flag)
                SELECT
                    symbol_id,
                    date,
                    CASE WHEN close = 0 THEN NULL
                        ELSE open * (base_close * total_return_index / close)
                    END AS adj_open,
                    CASE WHEN close = 0 THEN NULL
                        ELSE high * (base_close * total_return_index / close)
                    END AS adj_high,
                    CASE WHEN close = 0 THEN NULL
                        ELSE low * (base_close * total_return_index / close)
                    END AS adj_low,
                    CASE WHEN close = 0 THEN NULL
                        ELSE (base_close * total_return_index)
                    END AS adj_close,
                    volume * split_cum_factor AS adj_volume,
                    CASE WHEN close = 0 THEN NULL
                        ELSE (base_close * total_return_index / close)
                    END AS adj_factor,
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
            rows = self._resolve_rowcount(result, conn, "cur_prices_adjusted_daily")

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
            result = conn.execute(
                text("""
                INSERT INTO snap_universe_membership
                    (symbol_id, start_date, end_date, is_delisted, available_time, ingested_at)
                SELECT
                    s.symbol_id,
                    s.start_date,
                    s.end_date,
                    s.is_delisted,
                    (s.start_date + CAST(:close_time AS TIME)) AT TIME ZONE :exchange_tz
                        + (CAST(:delay_minutes AS INTEGER) * INTERVAL '1' MINUTE) AS available_time,
                    NOW() AS ingested_at
                FROM dim_symbol s
                WHERE s.start_date IS NOT NULL
                ON CONFLICT (symbol_id, start_date) DO UPDATE SET
                    end_date = EXCLUDED.end_date,
                    is_delisted = EXCLUDED.is_delisted,
                    available_time = EXCLUDED.available_time,
                    ingested_at = EXCLUDED.ingested_at
            """),
                {
                    "close_time": close_time,
                    "exchange_tz": exchange_tz,
                    "delay_minutes": delay_minutes,
                },
            )
            conn.commit()
            rows = self._resolve_rowcount(result, conn, "snap_universe_membership")

        logger.info(f"Transformed {rows} universe membership records")
        self._record_lineage(
            "dim_symbol", "snap_universe_membership", "transform_universe_membership"
        )
        return rows

    # ------------------------------------------------------------------
    # Stub transforms for new data sources (not yet implemented)
    # ------------------------------------------------------------------

    def transform_fundamentals(self) -> int:
        """Transform raw SEC fundamentals into curated quarterly metrics.

        Deduplicates by taking the latest filing per (ticker, metric,
        fiscal_period_end), excluding amendments.  ``available_time`` is
        derived from the SEC filing date plus a configurable delay.
        """
        logger.info("Transforming fundamentals...")
        base_delay = 60.0  # SEC EDGAR default lag in minutes
        latency_delay = self._latency_minutes("sec_edgar", base_delay)
        delay_minutes = max(base_delay, latency_delay)

        with self.db.engine.connect() as conn:
            result = conn.execute(
                text("""
                WITH ranked AS (
                    SELECT
                        r.ticker,
                        r.fiscal_period_end,
                        r.filing_date,
                        r.metric_name,
                        r.metric_value,
                        r.units,
                        ROW_NUMBER() OVER (
                            PARTITION BY r.ticker, r.metric_name, r.fiscal_period_end
                            ORDER BY r.filing_date DESC, r.filing_sequence DESC NULLS LAST
                        ) AS rn
                    FROM raw_sec_fundamentals r
                    WHERE COALESCE(r.is_amendment, false) = false
                )
                INSERT INTO cur_fundamentals_quarterly
                (symbol_id, fiscal_period_end, filing_date,
                 metric_name, metric_value, units,
                 event_time, available_time, time_quality,
                 ingested_at, data_quality_flag)
                SELECT
                    s.symbol_id,
                    rk.fiscal_period_end,
                    rk.filing_date,
                    rk.metric_name,
                    rk.metric_value,
                    rk.units,
                    rk.fiscal_period_end::TIMESTAMPTZ AS event_time,
                    (rk.filing_date::TIMESTAMPTZ
                     + INTERVAL '1 minute' * :delay) AS available_time,
                    'confirmed' AS time_quality,
                    NOW() AS ingested_at,
                    CASE WHEN rk.metric_value IS NULL
                         THEN 'missing_value' ELSE NULL
                    END AS data_quality_flag
                FROM ranked rk
                JOIN dim_symbol s ON rk.ticker = s.ticker
                WHERE rk.rn = 1
                ON CONFLICT (symbol_id, fiscal_period_end, metric_name) DO UPDATE SET
                    filing_date = EXCLUDED.filing_date,
                    metric_value = EXCLUDED.metric_value,
                    units = EXCLUDED.units,
                    available_time = EXCLUDED.available_time,
                    ingested_at = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag
            """),
                {"delay": delay_minutes},
            )
            conn.commit()
            rows = self._resolve_rowcount(result, conn, "cur_fundamentals_quarterly")

        logger.info(f"Transformed {rows} fundamental metrics")
        self._record_lineage(
            "raw_sec_fundamentals",
            "cur_fundamentals_quarterly",
            "transform_fundamentals",
        )
        return rows

    def transform_insider_trades(self) -> int:
        """Transform raw SEC Form-4 insider trades into curated table.

        Computes ``total_value`` (shares × price) and derives
        ``available_time`` from the SEC filing date.
        """
        logger.info("Transforming insider trades...")
        base_delay = 60.0
        latency_delay = self._latency_minutes("sec_edgar", base_delay)
        delay_minutes = max(base_delay, latency_delay)

        with self.db.engine.connect() as conn:
            result = conn.execute(
                text("""
                INSERT INTO cur_insider_trades
                (symbol_id, insider_name, insider_title,
                 transaction_date, transaction_type,
                 shares, price_per_share, total_value,
                 shares_after, ownership_type,
                 event_time, available_time, time_quality,
                 ingested_at, data_quality_flag)
                SELECT
                    s.symbol_id,
                    r.insider_name,
                    r.insider_title,
                    r.transaction_date,
                    r.transaction_type,
                    r.shares,
                    r.price_per_share,
                    r.shares * r.price_per_share AS total_value,
                    r.shares_after,
                    r.ownership_type,
                    r.transaction_date::TIMESTAMPTZ AS event_time,
                    (r.filing_date::TIMESTAMPTZ
                     + INTERVAL '1 minute' * :delay) AS available_time,
                    'confirmed' AS time_quality,
                    NOW() AS ingested_at,
                    CASE
                        WHEN r.shares IS NULL THEN 'missing_shares'
                        WHEN r.price_per_share IS NULL THEN 'missing_price'
                        ELSE NULL
                    END AS data_quality_flag
                FROM raw_sec_insider_trades r
                JOIN dim_symbol s ON r.ticker = s.ticker
                ON CONFLICT (symbol_id, insider_name, transaction_date,
                             transaction_type, shares) DO UPDATE SET
                    insider_title = EXCLUDED.insider_title,
                    price_per_share = EXCLUDED.price_per_share,
                    total_value = EXCLUDED.total_value,
                    shares_after = EXCLUDED.shares_after,
                    ownership_type = EXCLUDED.ownership_type,
                    available_time = EXCLUDED.available_time,
                    ingested_at = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag
            """),
                {"delay": delay_minutes},
            )
            conn.commit()
            rows = self._resolve_rowcount(result, conn, "cur_insider_trades")

        logger.info(f"Transformed {rows} insider trades")
        self._record_lineage(
            "raw_sec_insider_trades",
            "cur_insider_trades",
            "transform_insider_trades",
        )
        return rows

    def transform_institutional_holdings(self) -> int:
        """Transform raw SEC 13-F institutional holdings into curated table.

        Maps CUSIPs to symbols via ``settings.sec.cusip_mapping`` and
        computes ``pct_of_portfolio`` as the filer's market-value weight.
        """
        logger.info("Transforming institutional holdings...")
        base_delay = 60.0
        latency_delay = self._latency_minutes("sec_edgar", base_delay)
        delay_minutes = max(base_delay, latency_delay)

        # Build a temporary CUSIP→symbol_id lookup from settings
        cusip_map = self.settings.sec_edgar.cusip_mapping  # ticker → cusip
        # We need cusip → ticker (reversed)
        cusip_to_ticker = {v: k for k, v in cusip_map.items()}

        if not cusip_to_ticker:
            logger.warning("No CUSIP mapping configured; skipping 13-F transform")
            return 0

        with self.db.engine.connect() as conn:
            # Create a temp table with the CUSIP→ticker mapping
            conn.execute(
                text(
                    "CREATE TEMPORARY TABLE IF NOT EXISTS _tmp_cusip_map "
                    "(cusip VARCHAR(9) PRIMARY KEY, ticker VARCHAR(20))"
                )
            )
            conn.execute(text("DELETE FROM _tmp_cusip_map"))
            for cusip, ticker in cusip_to_ticker.items():
                conn.execute(
                    text("INSERT INTO _tmp_cusip_map (cusip, ticker) " "VALUES (:cusip, :ticker)"),
                    {"cusip": cusip, "ticker": ticker},
                )

            result = conn.execute(
                text("""
                WITH portfolio_totals AS (
                    SELECT
                        filer_cik,
                        report_date,
                        SUM(market_value) AS total_mv
                    FROM raw_sec_13f_holdings
                    GROUP BY filer_cik, report_date
                )
                INSERT INTO cur_institutional_holdings
                (symbol_id, filer_entity_id, filer_name, report_date,
                 market_value, shares_held, shares_type, put_call,
                 pct_of_portfolio,
                 event_time, available_time, time_quality,
                 ingested_at, data_quality_flag)
                SELECT
                    s.symbol_id,
                    NULL AS filer_entity_id,
                    r.filer_name,
                    r.report_date,
                    r.market_value,
                    r.shares_held,
                    r.shares_type,
                    r.put_call,
                    CASE WHEN pt.total_mv > 0
                         THEN r.market_value::NUMERIC / pt.total_mv
                         ELSE NULL
                    END AS pct_of_portfolio,
                    r.report_date::TIMESTAMPTZ AS event_time,
                    (r.filing_date::TIMESTAMPTZ
                     + INTERVAL '1 minute' * :delay) AS available_time,
                    'confirmed' AS time_quality,
                    NOW() AS ingested_at,
                    CASE WHEN r.market_value IS NULL OR r.market_value <= 0
                         THEN 'invalid_market_value'
                         ELSE NULL
                    END AS data_quality_flag
                FROM raw_sec_13f_holdings r
                JOIN _tmp_cusip_map cm ON r.cusip = cm.cusip
                JOIN dim_symbol s ON cm.ticker = s.ticker
                JOIN portfolio_totals pt
                    ON r.filer_cik = pt.filer_cik
                   AND r.report_date = pt.report_date
                ON CONFLICT (symbol_id, filer_name, report_date) DO UPDATE SET
                    market_value = EXCLUDED.market_value,
                    shares_held = EXCLUDED.shares_held,
                    shares_type = EXCLUDED.shares_type,
                    put_call = EXCLUDED.put_call,
                    pct_of_portfolio = EXCLUDED.pct_of_portfolio,
                    available_time = EXCLUDED.available_time,
                    ingested_at = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag
            """),
                {"delay": delay_minutes},
            )
            conn.commit()
            rows = self._resolve_rowcount(result, conn, "cur_institutional_holdings")

        logger.info(f"Transformed {rows} institutional holdings")
        self._record_lineage(
            "raw_sec_13f_holdings",
            "cur_institutional_holdings",
            "transform_institutional_holdings",
        )
        return rows

    def transform_options_summary(self) -> int:
        """Transform raw options chains into daily summary metrics.

        Aggregates per-strike rows into a single row per (symbol, date)
        with IV term-structure, put/call ratios, and skew.  Uses pandas
        for the bucketing logic since it requires non-trivial grouping
        by days-to-expiry.
        """
        logger.info("Transforming options summary...")

        if not self.db.table_exists("raw_options_chain"):
            logger.info("raw_options_chain table does not exist; skipping")
            return 0

        close_time = self.settings.prices.market_close_time
        exchange_tz = self.settings.prices.exchange_timezone
        base_delay = self.settings.prices.vendor_delay_minutes
        latency_delay = self._latency_minutes("options", base_delay)
        delay_minutes = max(base_delay, latency_delay)

        with self.db.engine.connect() as conn:
            # Read raw options + close prices for ATM identification
            df = pd.read_sql(
                text("""
                SELECT
                    r.ticker,
                    r.quote_date,
                    r.expiration,
                    r.strike,
                    r.option_type,
                    r.implied_volatility,
                    r.volume,
                    r.open_interest,
                    (r.expiration - r.quote_date) AS dte
                FROM raw_options_chain r
                WHERE r.implied_volatility IS NOT NULL
                  AND r.implied_volatility > 0
            """),
                conn,
            )

        if df.empty:
            logger.info("No options data to transform")
            return 0

        df["dte"] = df["dte"].dt.days if hasattr(df["dte"], "dt") else df["dte"]

        # Get latest close prices for ATM strike identification
        with self.db.engine.connect() as conn:
            prices_df = pd.read_sql(
                text("""
                SELECT s.ticker, p.date, p.close
                FROM cur_prices_ohlcv_daily p
                JOIN dim_symbol s ON p.symbol_id = s.symbol_id
            """),
                conn,
            )

        # Merge close prices to identify ATM
        df = df.merge(
            prices_df.rename(columns={"date": "quote_date"}),
            on=["ticker", "quote_date"],
            how="left",
        )

        results = []
        for (ticker, quote_date), grp in df.groupby(["ticker", "quote_date"]):
            calls = grp[grp["option_type"] == "call"]
            puts = grp[grp["option_type"] == "put"]

            total_call_vol = int(calls["volume"].sum()) if not calls.empty else 0
            total_put_vol = int(puts["volume"].sum()) if not puts.empty else 0
            total_call_oi = int(calls["open_interest"].sum()) if not calls.empty else 0
            total_put_oi = int(puts["open_interest"].sum()) if not puts.empty else 0

            pc_vol_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else None
            pc_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else None

            # IV by DTE bucket (mean IV of options within range)
            def _iv_bucket(lo: int, hi: int, _grp: pd.DataFrame = grp) -> float | None:
                mask = (_grp["dte"] >= lo) & (_grp["dte"] <= hi)
                bucket = _grp.loc[mask, "implied_volatility"]
                return float(bucket.mean()) if len(bucket) >= 2 else None

            iv_30d = _iv_bucket(20, 40)
            iv_60d = _iv_bucket(45, 75)
            iv_90d = _iv_bucket(75, 105)

            # ATM IV: strike closest to close price
            close_px = (
                grp["close"].iloc[0]
                if "close" in grp.columns and pd.notna(grp["close"].iloc[0])
                else None
            )
            iv_atm_call = None
            iv_atm_put = None
            if close_px is not None and close_px > 0:
                near_30 = grp[(grp["dte"] >= 20) & (grp["dte"] <= 40)].copy()
                if not near_30.empty:
                    near_30["strike_dist"] = (near_30["strike"] - close_px).abs()
                    atm_calls = near_30[near_30["option_type"] == "call"]
                    atm_puts = near_30[near_30["option_type"] == "put"]
                    if not atm_calls.empty:
                        iv_atm_call = float(
                            atm_calls.loc[atm_calls["strike_dist"].idxmin(), "implied_volatility"]
                        )
                    if not atm_puts.empty:
                        iv_atm_put = float(
                            atm_puts.loc[atm_puts["strike_dist"].idxmin(), "implied_volatility"]
                        )

            # 25-delta skew: OTM put IV - ATM call IV (approximate)
            skew_25d = None
            if close_px is not None and close_px > 0 and iv_atm_call is not None:
                otm_puts = grp[
                    (grp["option_type"] == "put")
                    & (grp["strike"] < close_px * 0.95)
                    & (grp["dte"] >= 20)
                    & (grp["dte"] <= 40)
                ]
                if not otm_puts.empty:
                    otm_put_iv = float(otm_puts["implied_volatility"].mean())
                    skew_25d = otm_put_iv - iv_atm_call

            iv_term_slope = (
                (iv_90d - iv_30d) / 60.0 if iv_30d is not None and iv_90d is not None else None
            )

            dq_flag = None
            if len(grp) < 10:
                dq_flag = "low_strike_coverage"

            results.append(
                {
                    "ticker": ticker,
                    "date": quote_date,
                    "iv_30d": iv_30d,
                    "iv_60d": iv_60d,
                    "iv_90d": iv_90d,
                    "iv_atm_call": iv_atm_call,
                    "iv_atm_put": iv_atm_put,
                    "put_call_volume_ratio": pc_vol_ratio,
                    "put_call_oi_ratio": pc_oi_ratio,
                    "total_call_volume": total_call_vol,
                    "total_put_volume": total_put_vol,
                    "total_call_oi": total_call_oi,
                    "total_put_oi": total_put_oi,
                    "skew_25d": skew_25d,
                    "iv_term_slope": iv_term_slope,
                    "data_quality_flag": dq_flag,
                }
            )

        if not results:
            logger.info("No options summaries computed")
            return 0

        res_df = pd.DataFrame(results)

        # Insert via SQL with symbol_id resolution
        rows = 0
        with self.db.engine.connect() as conn:
            for _, row in res_df.iterrows():
                r = conn.execute(
                    text("""
                    INSERT INTO cur_options_summary_daily
                    (symbol_id, date,
                     iv_30d, iv_60d, iv_90d, iv_atm_call, iv_atm_put,
                     put_call_volume_ratio, put_call_oi_ratio,
                     total_call_volume, total_put_volume,
                     total_call_oi, total_put_oi,
                     skew_25d, iv_term_slope,
                     event_time, available_time, time_quality,
                     ingested_at, data_quality_flag)
                    SELECT
                        s.symbol_id,
                        :date,
                        :iv_30d, :iv_60d, :iv_90d,
                        :iv_atm_call, :iv_atm_put,
                        :pc_vol, :pc_oi,
                        :call_vol, :put_vol, :call_oi, :put_oi,
                        :skew, :term_slope,
                        :date::TIMESTAMPTZ AS event_time,
                        (:date::TEXT || ' ' || :close_time)::TIMESTAMPTZ
                            AT TIME ZONE :tz AT TIME ZONE 'UTC'
                            + INTERVAL '1 minute' * :delay
                            AS available_time,
                        'confirmed' AS time_quality,
                        NOW() AS ingested_at,
                        :dq_flag
                    FROM dim_symbol s
                    WHERE s.ticker = :ticker
                    ON CONFLICT (symbol_id, date) DO UPDATE SET
                        iv_30d = EXCLUDED.iv_30d,
                        iv_60d = EXCLUDED.iv_60d,
                        iv_90d = EXCLUDED.iv_90d,
                        iv_atm_call = EXCLUDED.iv_atm_call,
                        iv_atm_put = EXCLUDED.iv_atm_put,
                        put_call_volume_ratio = EXCLUDED.put_call_volume_ratio,
                        put_call_oi_ratio = EXCLUDED.put_call_oi_ratio,
                        total_call_volume = EXCLUDED.total_call_volume,
                        total_put_volume = EXCLUDED.total_put_volume,
                        total_call_oi = EXCLUDED.total_call_oi,
                        total_put_oi = EXCLUDED.total_put_oi,
                        skew_25d = EXCLUDED.skew_25d,
                        iv_term_slope = EXCLUDED.iv_term_slope,
                        ingested_at = EXCLUDED.ingested_at,
                        data_quality_flag = EXCLUDED.data_quality_flag
                """),
                    {
                        "ticker": row["ticker"],
                        "date": row["date"],
                        "iv_30d": row.get("iv_30d"),
                        "iv_60d": row.get("iv_60d"),
                        "iv_90d": row.get("iv_90d"),
                        "iv_atm_call": row.get("iv_atm_call"),
                        "iv_atm_put": row.get("iv_atm_put"),
                        "pc_vol": row.get("put_call_volume_ratio"),
                        "pc_oi": row.get("put_call_oi_ratio"),
                        "call_vol": row.get("total_call_volume"),
                        "put_vol": row.get("total_put_volume"),
                        "call_oi": row.get("total_call_oi"),
                        "put_oi": row.get("total_put_oi"),
                        "skew": row.get("skew_25d"),
                        "term_slope": row.get("iv_term_slope"),
                        "close_time": close_time,
                        "tz": exchange_tz,
                        "delay": delay_minutes,
                        "dq_flag": row.get("data_quality_flag"),
                    },
                )
                rows += r.rowcount if r.rowcount > 0 else 0
            conn.commit()

        logger.info(f"Transformed {rows} options summary records")
        self._record_lineage(
            "raw_options_chain",
            "cur_options_summary_daily",
            "transform_options_summary",
        )
        return rows

    def transform_earnings(self) -> int:
        """Transform raw earnings calendar into curated earnings events.

        Computes EPS/revenue surprise metrics and sets ``available_time``
        based on whether the report was before-market-open or after-close.
        """
        logger.info("Transforming earnings events...")
        close_time = self.settings.prices.market_close_time
        exchange_tz = self.settings.prices.exchange_timezone
        base_delay = self.settings.prices.vendor_delay_minutes
        latency_delay = self._latency_minutes("earnings", base_delay)
        delay_minutes = max(base_delay, latency_delay)

        with self.db.engine.connect() as conn:
            result = conn.execute(
                text("""
                INSERT INTO cur_earnings_events
                (symbol_id, report_date, fiscal_quarter_end,
                 eps_estimate, eps_actual, eps_surprise, eps_surprise_pct,
                 revenue_estimate, revenue_actual,
                 revenue_surprise, revenue_surprise_pct,
                 report_time,
                 event_time, available_time, time_quality,
                 ingested_at, data_quality_flag)
                SELECT
                    s.symbol_id,
                    r.report_date,
                    r.fiscal_quarter_end,
                    r.eps_estimate,
                    r.eps_actual,
                    r.eps_actual - r.eps_estimate AS eps_surprise,
                    CASE WHEN r.eps_estimate IS NOT NULL
                              AND ABS(r.eps_estimate) > 0
                         THEN (r.eps_actual - r.eps_estimate)
                              / ABS(r.eps_estimate)
                         ELSE NULL
                    END AS eps_surprise_pct,
                    r.revenue_estimate,
                    r.revenue_actual,
                    r.revenue_actual - r.revenue_estimate AS revenue_surprise,
                    CASE WHEN r.revenue_estimate IS NOT NULL
                              AND ABS(r.revenue_estimate) > 0
                         THEN (r.revenue_actual - r.revenue_estimate)
                              / ABS(r.revenue_estimate)
                         ELSE NULL
                    END AS revenue_surprise_pct,
                    r.report_time,
                    r.report_date::TIMESTAMPTZ AS event_time,
                    CASE
                        WHEN LOWER(r.report_time) = 'bmo'
                            THEN (r.report_date::TEXT || ' 09:30:00')::TIMESTAMPTZ
                                 AT TIME ZONE :tz AT TIME ZONE 'UTC'
                        WHEN LOWER(r.report_time) = 'amc'
                            THEN (r.report_date::TEXT || ' ' || :close_time)::TIMESTAMPTZ
                                 AT TIME ZONE :tz AT TIME ZONE 'UTC'
                        ELSE (r.report_date::TEXT || ' ' || :close_time)::TIMESTAMPTZ
                             AT TIME ZONE :tz AT TIME ZONE 'UTC'
                             + INTERVAL '1 minute' * :delay
                    END AS available_time,
                    CASE WHEN r.report_time IS NOT NULL
                         THEN 'confirmed' ELSE 'assumed'
                    END AS time_quality,
                    NOW() AS ingested_at,
                    CASE
                        WHEN r.eps_actual IS NULL THEN 'pending'
                        WHEN r.eps_estimate IS NULL THEN 'no_estimate'
                        ELSE NULL
                    END AS data_quality_flag
                FROM raw_earnings_calendar r
                JOIN dim_symbol s ON r.ticker = s.ticker
                ON CONFLICT (symbol_id, report_date) DO UPDATE SET
                    fiscal_quarter_end = EXCLUDED.fiscal_quarter_end,
                    eps_estimate = EXCLUDED.eps_estimate,
                    eps_actual = EXCLUDED.eps_actual,
                    eps_surprise = EXCLUDED.eps_surprise,
                    eps_surprise_pct = EXCLUDED.eps_surprise_pct,
                    revenue_estimate = EXCLUDED.revenue_estimate,
                    revenue_actual = EXCLUDED.revenue_actual,
                    revenue_surprise = EXCLUDED.revenue_surprise,
                    revenue_surprise_pct = EXCLUDED.revenue_surprise_pct,
                    report_time = EXCLUDED.report_time,
                    available_time = EXCLUDED.available_time,
                    time_quality = EXCLUDED.time_quality,
                    ingested_at = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag
            """),
                {"tz": exchange_tz, "close_time": close_time, "delay": delay_minutes},
            )
            conn.commit()
            rows = self._resolve_rowcount(result, conn, "cur_earnings_events")

        logger.info(f"Transformed {rows} earnings events")
        self._record_lineage(
            "raw_earnings_calendar",
            "cur_earnings_events",
            "transform_earnings",
        )
        return rows

    def transform_short_interest(self) -> int:
        """Transform raw short interest into curated table.

        Computes change-over-change metrics using ``LAG`` and sets
        ``available_time`` to settlement date + 14 days (FINRA
        publication delay approximation).
        """
        logger.info("Transforming short interest...")

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                WITH base AS (
                    SELECT
                        s.symbol_id,
                        r.settlement_date,
                        r.short_interest,
                        r.avg_daily_volume,
                        r.days_to_cover,
                        LAG(r.short_interest) OVER (
                            PARTITION BY s.symbol_id
                            ORDER BY r.settlement_date
                        ) AS prev_short_interest
                    FROM raw_short_interest r
                    JOIN dim_symbol s ON r.ticker = s.ticker
                )
                INSERT INTO cur_short_interest
                (symbol_id, settlement_date,
                 short_interest, avg_daily_volume, days_to_cover,
                 short_pct_float,
                 short_interest_change, short_interest_change_pct,
                 event_time, available_time, time_quality,
                 ingested_at, data_quality_flag)
                SELECT
                    b.symbol_id,
                    b.settlement_date,
                    b.short_interest,
                    b.avg_daily_volume,
                    b.days_to_cover,
                    NULL AS short_pct_float,
                    b.short_interest - b.prev_short_interest
                        AS short_interest_change,
                    CASE WHEN b.prev_short_interest IS NOT NULL
                              AND b.prev_short_interest > 0
                         THEN (b.short_interest - b.prev_short_interest)::NUMERIC
                              / b.prev_short_interest
                         ELSE NULL
                    END AS short_interest_change_pct,
                    b.settlement_date::TIMESTAMPTZ AS event_time,
                    (b.settlement_date + INTERVAL '14 days')::TIMESTAMPTZ
                        AS available_time,
                    'assumed' AS time_quality,
                    NOW() AS ingested_at,
                    CASE WHEN b.short_interest < 0
                         THEN 'negative_short_interest'
                         ELSE NULL
                    END AS data_quality_flag
                FROM base b
                ON CONFLICT (symbol_id, settlement_date) DO UPDATE SET
                    short_interest = EXCLUDED.short_interest,
                    avg_daily_volume = EXCLUDED.avg_daily_volume,
                    days_to_cover = EXCLUDED.days_to_cover,
                    short_interest_change = EXCLUDED.short_interest_change,
                    short_interest_change_pct = EXCLUDED.short_interest_change_pct,
                    available_time = EXCLUDED.available_time,
                    ingested_at = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag
            """))
            conn.commit()
            rows = self._resolve_rowcount(result, conn, "cur_short_interest")

        logger.info(f"Transformed {rows} short interest records")
        self._record_lineage(
            "raw_short_interest",
            "cur_short_interest",
            "transform_short_interest",
        )
        return rows

    def transform_etf_flows(self) -> int:
        """Transform raw ETF fund-flow data into curated daily flows.

        Computes ``flow_pct_aum`` and rolling 5-day / 20-day flow sums.
        ``available_time`` is set to T+1 (next calendar day).
        """
        logger.info("Transforming ETF flows...")

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                WITH base AS (
                    SELECT
                        s.symbol_id,
                        r.date,
                        r.fund_flow,
                        r.aum,
                        CASE WHEN r.aum IS NOT NULL AND r.aum > 0
                             THEN r.fund_flow / r.aum
                             ELSE NULL
                        END AS flow_pct_aum,
                        SUM(r.fund_flow) OVER (
                            PARTITION BY s.symbol_id
                            ORDER BY r.date
                            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                        ) AS flow_5d_sum,
                        SUM(r.fund_flow) OVER (
                            PARTITION BY s.symbol_id
                            ORDER BY r.date
                            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                        ) AS flow_20d_sum
                    FROM raw_etf_flows r
                    JOIN dim_symbol s ON r.ticker = s.ticker
                )
                INSERT INTO cur_etf_flows_daily
                (symbol_id, date, fund_flow, aum,
                 flow_pct_aum, flow_5d_sum, flow_20d_sum,
                 event_time, available_time, time_quality,
                 ingested_at, data_quality_flag)
                SELECT
                    b.symbol_id,
                    b.date,
                    b.fund_flow,
                    b.aum,
                    b.flow_pct_aum,
                    b.flow_5d_sum,
                    b.flow_20d_sum,
                    b.date::TIMESTAMPTZ AS event_time,
                    (b.date + INTERVAL '1 day')::TIMESTAMPTZ AS available_time,
                    'assumed' AS time_quality,
                    NOW() AS ingested_at,
                    CASE
                        WHEN b.aum IS NULL OR b.aum <= 0
                            THEN 'invalid_aum'
                        WHEN b.fund_flow IS NULL
                            THEN 'missing_flow'
                        ELSE NULL
                    END AS data_quality_flag
                FROM base b
                ON CONFLICT (symbol_id, date) DO UPDATE SET
                    fund_flow = EXCLUDED.fund_flow,
                    aum = EXCLUDED.aum,
                    flow_pct_aum = EXCLUDED.flow_pct_aum,
                    flow_5d_sum = EXCLUDED.flow_5d_sum,
                    flow_20d_sum = EXCLUDED.flow_20d_sum,
                    available_time = EXCLUDED.available_time,
                    ingested_at = EXCLUDED.ingested_at,
                    data_quality_flag = EXCLUDED.data_quality_flag
            """))
            conn.commit()
            rows = self._resolve_rowcount(result, conn, "cur_etf_flows_daily")

        logger.info(f"Transformed {rows} ETF flow records")
        self._record_lineage(
            "raw_etf_flows",
            "cur_etf_flows_daily",
            "transform_etf_flows",
        )
        return rows

    def transform_cftc_cot(self) -> int:
        """Transform raw CFTC COT data into curated table.

        Computes net positioning and percentage-of-open-interest metrics.
        ``available_time`` is set to Friday 19:30 UTC (3:30 PM ET) of the
        report week, reflecting the CFTC's publication schedule.
        """
        logger.info("Transforming CFTC COT data...")

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                    INSERT INTO cur_cftc_cot
                    (commodity_code, report_date,
                     commercial_net, noncommercial_net, nonreportable_net,
                     open_interest, commercial_pct_oi, noncommercial_pct_oi,
                     event_time, available_time, time_quality,
                     ingested_at, data_quality_flag)
                    SELECT
                        r.commodity_code,
                        r.report_date,
                        r.commercial_long - r.commercial_short AS commercial_net,
                        r.noncommercial_long - r.noncommercial_short AS noncommercial_net,
                        r.nonreportable_long - r.nonreportable_short AS nonreportable_net,
                        r.open_interest,
                        CASE WHEN r.open_interest > 0
                             THEN (r.commercial_long - r.commercial_short)::NUMERIC
                                  / r.open_interest
                             ELSE NULL
                        END AS commercial_pct_oi,
                        CASE WHEN r.open_interest > 0
                             THEN (r.noncommercial_long - r.noncommercial_short)::NUMERIC
                                  / r.open_interest
                             ELSE NULL
                        END AS noncommercial_pct_oi,
                        r.report_date::TIMESTAMPTZ AS event_time,
                        (r.report_date + INTERVAL '3 days'
                         + INTERVAL '19 hours 30 minutes')::TIMESTAMPTZ AS available_time,
                        'confirmed' AS time_quality,
                        NOW() AS ingested_at,
                        CASE
                            WHEN r.open_interest IS NULL OR r.open_interest <= 0
                                THEN 'invalid_open_interest'
                            WHEN r.commercial_long IS NULL OR r.commercial_short IS NULL
                                THEN 'missing_positions'
                            ELSE NULL
                        END AS data_quality_flag
                    FROM raw_cftc_cot r
                    ON CONFLICT (commodity_code, report_date) DO UPDATE SET
                        commercial_net = EXCLUDED.commercial_net,
                        noncommercial_net = EXCLUDED.noncommercial_net,
                        nonreportable_net = EXCLUDED.nonreportable_net,
                        open_interest = EXCLUDED.open_interest,
                        commercial_pct_oi = EXCLUDED.commercial_pct_oi,
                        noncommercial_pct_oi = EXCLUDED.noncommercial_pct_oi,
                        available_time = EXCLUDED.available_time,
                        ingested_at = EXCLUDED.ingested_at,
                        data_quality_flag = EXCLUDED.data_quality_flag
                """))
            conn.commit()
            rows = self._resolve_rowcount(result, conn, "cur_cftc_cot")

        logger.info(f"Transformed {rows} CFTC COT records")
        self._record_lineage("raw_cftc_cot", "cur_cftc_cot", "transform_cftc_cot")
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
        # New data source transforms (call only if implemented)
        for name in (
            "fundamentals",
            "insider_trades",
            "institutional_holdings",
            "options_summary",
            "earnings",
            "short_interest",
            "etf_flows",
            "cftc_cot",
        ):
            method = getattr(self, f"transform_{name}", None)
            if method is not None:
                try:
                    results[name] = method()
                except Exception as exc:
                    logger.warning(f"transform_{name} failed: {exc}")
                    results[name] = 0
        return results
