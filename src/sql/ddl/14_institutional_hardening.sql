-- Institutional hardening migrations

-- Raw layer: ensure columns exist
ALTER TABLE raw_prices_ohlcv ADD COLUMN IF NOT EXISTS split_ratio VARCHAR(20);
ALTER TABLE raw_prices_ohlcv ADD COLUMN IF NOT EXISTS dividend NUMERIC DEFAULT 0;
ALTER TABLE raw_fred_observations ADD COLUMN IF NOT EXISTS realtime_start DATE;
ALTER TABLE raw_fred_observations ADD COLUMN IF NOT EXISTS realtime_end DATE;

-- Raw layer: orderbook snapshots
CREATE TABLE IF NOT EXISTS raw_polymarket_orderbook_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venue_market_id VARCHAR(100) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    best_bid NUMERIC,
    best_ask NUMERIC,
    spread NUMERIC,
    bids JSONB,
    asks JSONB,
    raw_data JSONB NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_raw_polymarket_ob_market_ts
    ON raw_polymarket_orderbook_snapshots(venue_market_id, ts);
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_polymarket_ob_unique
    ON raw_polymarket_orderbook_snapshots(venue_market_id, ts);

-- Raw layer: unique indexes for idempotency
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_fred_unique
    ON raw_fred_observations(series_code, observation_date, realtime_start);
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_gdelt_unique
    ON raw_gdelt_events(gdelt_event_id)
    WHERE gdelt_event_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_polymarket_markets_unique
    ON raw_polymarket_markets(venue_market_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_polymarket_prices_unique
    ON raw_polymarket_prices(venue_market_id, ts, outcome);
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_polymarket_trades_unique
    ON raw_polymarket_trades(trade_id)
    WHERE trade_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_prices_unique
    ON raw_prices_ohlcv(ticker, date);

-- Curated layer: add ingest/data quality columns
ALTER TABLE cur_prices_ohlcv_daily ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_prices_ohlcv_daily ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_corporate_actions ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_corporate_actions ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_fundamentals_quarterly ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_fundamentals_quarterly ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_macro_observations ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_macro_observations ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_macro_releases ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_macro_releases ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_news_items ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_news_items ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_world_events ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_world_events ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_contract_state_daily ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_contract_state_daily ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_contract_prices ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_contract_prices ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_contract_orderbook_snapshots ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_contract_orderbook_snapshots ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_contract_trades ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_contract_trades ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_contract_resolution ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_contract_resolution ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);
ALTER TABLE cur_factor_returns ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
ALTER TABLE cur_factor_returns ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(50);

-- Update time_quality checks to include 'inferred'
DO $$
DECLARE
    tbl text;
    conname text;
    tbls text[] := ARRAY[
        'cur_prices_ohlcv_daily',
        'cur_corporate_actions',
        'cur_fundamentals_quarterly',
        'cur_macro_observations',
        'cur_macro_releases',
        'cur_news_items',
        'cur_world_events',
        'cur_contract_state_daily',
        'cur_contract_prices',
        'cur_contract_orderbook_snapshots',
        'cur_contract_trades',
        'cur_contract_resolution',
        'cur_factor_returns'
    ];
BEGIN
    FOREACH tbl IN ARRAY tbls LOOP
        SELECT con.conname INTO conname
        FROM pg_constraint con
        WHERE con.conrelid = tbl::regclass
          AND con.contype = 'c'
          AND pg_get_constraintdef(con.oid) LIKE '%time_quality%';
        IF conname IS NOT NULL THEN
            EXECUTE format('ALTER TABLE %I DROP CONSTRAINT %I', tbl, conname);
        END IF;
        EXECUTE format(
            'ALTER TABLE %I ADD CONSTRAINT %I_time_quality_check CHECK (time_quality IN (''assumed'',''confirmed'',''inferred''))',
            tbl,
            tbl
        );
    END LOOP;
END$$;

-- Adjusted prices (point-in-time)
CREATE TABLE IF NOT EXISTS cur_prices_adjusted_daily (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    date DATE NOT NULL,
    adj_open NUMERIC,
    adj_high NUMERIC,
    adj_low NUMERIC,
    adj_close NUMERIC,
    adj_volume NUMERIC,
    adj_factor NUMERIC NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, date)
);

CREATE INDEX IF NOT EXISTS idx_cur_prices_adj_available_time ON cur_prices_adjusted_daily(available_time);
CREATE INDEX IF NOT EXISTS idx_cur_prices_adj_date ON cur_prices_adjusted_daily(date);

-- Universe membership snapshots
CREATE TABLE IF NOT EXISTS snap_universe_membership (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    start_date DATE NOT NULL,
    end_date DATE,
    is_delisted BOOLEAN NOT NULL DEFAULT FALSE,
    available_time TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, start_date)
);

CREATE INDEX IF NOT EXISTS idx_snap_universe_available_time ON snap_universe_membership(available_time);
CREATE INDEX IF NOT EXISTS idx_snap_universe_end_date ON snap_universe_membership(end_date);
