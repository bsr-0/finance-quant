"""Transform raw data into curated tables."""

import logging
from uuid import UUID

from sqlalchemy import text

from pipeline.db import get_db_manager
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)

# A symbol with no trades for this many days is considered delisted.
_DELISTING_GAP_DAYS = 30


class CuratedTransformer:
    """Transform raw data into curated, deduplicated tables."""

    def __init__(self):
        self.db = get_db_manager()

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
        settings = get_settings()
        urls = {
            "fred": settings.fred.base_url,
            "gdelt": settings.gdelt.base_url,
            "polymarket": settings.polymarket.base_url,
            "prices": "https://finance.yahoo.com",
        }
        return urls.get(source_name, "")

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
                    (provider_series_code, name, frequency, source_id)
                SELECT DISTINCT
                    r.series_code,
                    r.series_code AS name,
                    'monthly'     AS frequency,
                    :source_id
                FROM raw_fred_observations r
                LEFT JOIN dim_macro_series d
                    ON r.series_code = d.provider_series_code
                WHERE d.series_id IS NULL
            """),
                {"source_id": source_id},
            )
            conn.commit()

        # Transform observations.
        # available_time = realtime_start (when FRED first published the value)
        # event_time     = observation_date (period the value refers to)
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_macro_observations
                    (series_id, period_end, value, revision_id,
                     event_time, available_time, time_quality)
                SELECT
                    s.series_id,
                    r.observation_date                                AS period_end,
                    r.value,
                    COALESCE(r.realtime_start::text, 'initial')       AS revision_id,
                    r.observation_date::timestamptz                    AS event_time,
                    COALESCE(
                        r.realtime_start::timestamptz,
                        r.extracted_at
                    )                                                 AS available_time,
                    CASE
                        WHEN r.realtime_start IS NOT NULL THEN 'confirmed'
                        ELSE 'assumed'
                    END                                               AS time_quality
                FROM raw_fred_observations r
                JOIN dim_macro_series s
                    ON r.series_code = s.provider_series_code
                ON CONFLICT (series_id, period_end, revision_id) DO UPDATE SET
                    value          = EXCLUDED.value,
                    available_time = EXCLUDED.available_time,
                    time_quality   = EXCLUDED.time_quality,
                    updated_at     = NOW()
            """))
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} macro observations")
        return rows

    def transform_world_events(self) -> int:
        """Transform raw GDELT data to curated world events."""
        logger.info("Transforming world events...")

        source_id = self._get_source_id("gdelt")

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_world_events
                (source_id, gdelt_event_id, event_type, event_time, available_time,
                 location, actors, themes, tone_score, sentiment_positive,
                 sentiment_negative, time_quality)
                SELECT
                    :source_id,
                    (r.raw_data::json->>'GLOBALEVENTID')::bigint AS gdelt_event_id,
                    r.raw_data::json->>'EventCode'               AS event_type,
                    (r.raw_data::json->>'SQLDATE')::date::timestamptz AS event_time,
                    r.extracted_at                                AS available_time,
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
                    NULL                                          AS sentiment_positive,
                    NULL                                          AS sentiment_negative,
                    'assumed'                                     AS time_quality
                FROM raw_gdelt_events r
                LEFT JOIN cur_world_events c
                    ON (r.raw_data::json->>'GLOBALEVENTID')::bigint = c.gdelt_event_id
                WHERE c.event_id IS NULL
                ON CONFLICT DO NOTHING
            """),
                {"source_id": source_id},
            )
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} world events")
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
        return rows

    def transform_contract_prices(self) -> int:
        """Transform raw Polymarket prices to curated contract prices."""
        logger.info("Transforming contract prices...")

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_contract_prices
                (contract_id, ts, outcome, price_raw, price_normalized,
                 event_time, available_time, time_quality)
                SELECT
                    c.contract_id,
                    r.ts,
                    COALESCE(r.outcome, 'YES') AS outcome,
                    r.price                    AS price_raw,
                    r.price / 100.0            AS price_normalized,
                    r.ts                       AS event_time,
                    r.extracted_at             AS available_time,
                    'assumed'                  AS time_quality
                FROM raw_polymarket_prices r
                JOIN dim_contract c ON r.venue_market_id = c.venue_market_id
                ON CONFLICT (contract_id, ts, outcome) DO UPDATE SET
                    price_raw        = EXCLUDED.price_raw,
                    price_normalized = EXCLUDED.price_normalized,
                    available_time   = EXCLUDED.available_time
            """))
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} contract prices")
        return rows

    def transform_contract_trades(self) -> int:
        """Transform raw Polymarket trades to curated contract trades."""
        logger.info("Transforming contract trades...")

        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_contract_trades
                (contract_id, trade_id, ts, price, size, side,
                 event_time, available_time, time_quality)
                SELECT
                    c.contract_id,
                    r.trade_id,
                    r.ts,
                    r.price,
                    r.size,
                    r.side,
                    r.ts           AS event_time,
                    r.extracted_at AS available_time,
                    'assumed'      AS time_quality
                FROM raw_polymarket_trades r
                JOIN dim_contract c ON r.venue_market_id = c.venue_market_id
                ON CONFLICT (contract_id, trade_id) DO NOTHING
            """)
            )
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} contract trades")
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
                     event_time, available_time, time_quality)
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
                    (r.date || ' 16:00:00')::timestamptz AS event_time,
                    r.extracted_at     AS available_time,
                    'confirmed'        AS time_quality
                FROM raw_prices_ohlcv r
                JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE r.split_ratio IS NOT NULL
                ON CONFLICT (symbol_id, action_type, action_date) DO NOTHING
            """))
            conn.commit()
            splits = result.rowcount
            if splits:
                logger.info(f"Inserted {splits} split actions")

        # ------ 4. Populate cur_corporate_actions (dividends) ------
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_corporate_actions
                    (symbol_id, action_type, action_date, amount,
                     event_time, available_time, time_quality)
                SELECT
                    s.symbol_id,
                    'dividend'         AS action_type,
                    r.date             AS action_date,
                    r.dividend         AS amount,
                    (r.date || ' 16:00:00')::timestamptz AS event_time,
                    r.extracted_at     AS available_time,
                    'confirmed'        AS time_quality
                FROM raw_prices_ohlcv r
                JOIN dim_symbol s ON r.ticker = s.ticker
                WHERE r.dividend IS NOT NULL AND r.dividend > 0
                ON CONFLICT (symbol_id, action_type, action_date) DO NOTHING
            """))
            conn.commit()
            divs = result.rowcount
            if divs:
                logger.info(f"Inserted {divs} dividend actions")

        # ------ 5. Transform prices ------
        with self.db.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO cur_prices_ohlcv_daily
                    (symbol_id, date, open, high, low, close, adj_close, volume,
                     event_time, available_time, time_quality)
                SELECT
                    s.symbol_id,
                    r.date,
                    r.open,
                    r.high,
                    r.low,
                    r.close,
                    r.adj_close,
                    r.volume,
                    (r.date || ' 16:00:00')::timestamptz AS event_time,
                    r.extracted_at                        AS available_time,
                    'assumed'                             AS time_quality
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
                    updated_at     = NOW()
            """))
            conn.commit()
            rows = result.rowcount

        logger.info(f"Transformed {rows} OHLCV records")
        return rows

    # ------------------------------------------------------------------

    def transform_all(self) -> dict:
        """Run all transformations."""
        results = {}
        results["macro_observations"] = self.transform_macro_observations()
        results["world_events"] = self.transform_world_events()
        results["contracts"] = self.transform_contracts()
        results["contract_prices"] = self.transform_contract_prices()
        results["contract_trades"] = self.transform_contract_trades()
        results["prices_ohlcv"] = self.transform_prices_ohlcv()
        return results
