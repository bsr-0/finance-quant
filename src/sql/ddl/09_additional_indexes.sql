-- Additional performance indexes for common query patterns

-- Time-series indexes for efficient as-of queries
CREATE INDEX IF NOT EXISTS idx_cur_prices_asof 
    ON cur_prices_ohlcv_daily(symbol_id, available_time, date);

CREATE INDEX IF NOT EXISTS idx_cur_contract_prices_asof 
    ON cur_contract_prices(contract_id, available_time, ts);

CREATE INDEX IF NOT EXISTS idx_cur_contract_trades_asof 
    ON cur_contract_trades(contract_id, available_time, ts);

-- Composite indexes for common filter patterns
CREATE INDEX IF NOT EXISTS idx_cur_macro_series_country_freq 
    ON dim_macro_series(country, frequency) 
    WHERE country IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_dim_symbol_asset_delisted 
    ON dim_symbol(asset_class, is_delisted);

CREATE INDEX IF NOT EXISTS idx_dim_contract_status_venue 
    ON dim_contract(status, venue);

-- Partial indexes for active data
CREATE INDEX IF NOT EXISTS idx_cur_contract_active_prices 
    ON cur_contract_prices(contract_id, ts DESC) 
    WHERE price_normalized > 0;

-- GIN indexes for JSONB columns (if frequently queried)
CREATE INDEX IF NOT EXISTS idx_dim_symbol_external_ids 
    ON dim_symbol USING GIN(external_ids);

CREATE INDEX IF NOT EXISTS idx_dim_contract_outcomes 
    ON dim_contract USING GIN(outcomes);

-- Indexes for event-based queries
CREATE INDEX IF NOT EXISTS idx_cur_world_events_time_type 
    ON cur_world_events(event_time, event_type);

CREATE INDEX IF NOT EXISTS idx_cur_world_events_tone 
    ON cur_world_events(tone_score) 
    WHERE tone_score IS NOT NULL;

-- Indexes for news queries
CREATE INDEX IF NOT EXISTS idx_cur_news_entities 
    ON cur_news_items USING GIN(entities);

-- Indexes for snapshot queries
CREATE INDEX IF NOT EXISTS idx_snap_contract_features_contract_ts 
    ON snap_contract_features(contract_id, asof_ts DESC);

CREATE INDEX IF NOT EXISTS idx_snap_contract_features_ts 
    ON snap_contract_features(asof_ts DESC);

-- Covering indexes for common queries (include frequently accessed columns)
CREATE INDEX IF NOT EXISTS idx_cur_prices_covering 
    ON cur_prices_ohlcv_daily(symbol_id, date) 
    INCLUDE (open, high, low, close, volume);

-- Indexes for data quality checks
CREATE INDEX IF NOT EXISTS idx_cur_prices_time_quality 
    ON cur_prices_ohlcv_daily(time_quality) 
    WHERE time_quality = 'assumed';

-- BRIN indexes for very large time-series tables (space-efficient)
CREATE INDEX IF NOT EXISTS idx_raw_fred_observations_brin 
    ON raw_fred_observations USING BRIN(observation_date);

CREATE INDEX IF NOT EXISTS idx_raw_gdelt_events_brin 
    ON raw_gdelt_events USING BRIN(event_date);

-- Indexes for pipeline runs
CREATE INDEX IF NOT EXISTS idx_meta_pipeline_runs_time 
    ON meta_pipeline_runs(started_at DESC, finished_at);

-- Function to analyze all tables for query optimization
CREATE OR REPLACE FUNCTION analyze_all_tables()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN 
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE schemaname = 'public'
    LOOP
        EXECUTE format('ANALYZE %I.%I', r.schemaname, r.tablename);
    END LOOP;
END;
$$;
