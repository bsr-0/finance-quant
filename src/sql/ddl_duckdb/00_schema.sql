-- Consolidated DuckDB schema for Market Data Warehouse
-- Single file replaces 23 PostgreSQL DDL files. No server required.
--
-- Differences from PostgreSQL schema:
--   JSONB      -> JSON
--   BIGSERIAL  -> BIGINT (auto-generated via sequences)
--   gen_random_uuid() -> uuid()
--   No partitioning (DuckDB handles large tables natively)
--   No GIN/BRIN indexes (DuckDB uses zonemap indexes automatically)
--   No plpgsql functions
--   TIMESTAMPTZ -> TIMESTAMPTZ (supported natively)

-- Sequences for auto-incrementing IDs (replaces BIGSERIAL)
CREATE SEQUENCE IF NOT EXISTS raw_sec_fundamentals_seq;
CREATE SEQUENCE IF NOT EXISTS raw_sec_insider_trades_seq;
CREATE SEQUENCE IF NOT EXISTS raw_sec_13f_holdings_seq;
CREATE SEQUENCE IF NOT EXISTS raw_options_chain_seq;
CREATE SEQUENCE IF NOT EXISTS raw_earnings_calendar_seq;
CREATE SEQUENCE IF NOT EXISTS raw_reddit_posts_seq;
CREATE SEQUENCE IF NOT EXISTS raw_short_interest_seq;
CREATE SEQUENCE IF NOT EXISTS raw_etf_flows_seq;

-- ============================================================
-- META TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS meta_pipeline_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid(),
    started_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    finished_at TIMESTAMPTZ,
    git_sha VARCHAR(40),
    pipeline_name VARCHAR(100) NOT NULL,
    params JSON,
    status VARCHAR(20) NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    row_counts JSON,
    errors TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

-- ============================================================
-- DIMENSION TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS dim_source (
    source_id UUID PRIMARY KEY DEFAULT uuid(),
    name VARCHAR(50) NOT NULL UNIQUE,
    type VARCHAR(20) NOT NULL CHECK (type IN ('api', 'files', 'scrape')),
    base_url VARCHAR(500),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS dim_calendar_market (
    cal_id UUID PRIMARY KEY DEFAULT uuid(),
    name VARCHAR(50) NOT NULL UNIQUE,
    timezone VARCHAR(50) NOT NULL DEFAULT 'UTC',
    trading_days TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS dim_entity (
    entity_id UUID PRIMARY KEY DEFAULT uuid(),
    entity_type VARCHAR(30) NOT NULL CHECK (entity_type IN ('company', 'country', 'person', 'indicator', 'topic', 'organization')),
    name VARCHAR(200) NOT NULL,
    aliases JSON,
    external_ids JSON,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS dim_symbol (
    symbol_id UUID PRIMARY KEY DEFAULT uuid(),
    ticker VARCHAR(50) NOT NULL,
    exchange VARCHAR(50),
    asset_class VARCHAR(20) NOT NULL CHECK (asset_class IN ('equity', 'etf', 'index', 'fx', 'commodity', 'crypto')),
    currency VARCHAR(10) NOT NULL DEFAULT 'USD',
    start_date DATE,
    end_date DATE,
    is_delisted BOOLEAN NOT NULL DEFAULT FALSE,
    external_ids JSON,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(ticker, exchange)
);

CREATE TABLE IF NOT EXISTS dim_macro_series (
    series_id UUID PRIMARY KEY DEFAULT uuid(),
    provider_series_code VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    units VARCHAR(50),
    frequency VARCHAR(20) CHECK (frequency IN ('daily', 'weekly', 'monthly', 'quarterly', 'annual')),
    country VARCHAR(10),
    source_id UUID REFERENCES dim_source(source_id),
    release_time TIME,
    release_timezone VARCHAR(50),
    release_day_offset INTEGER,
    release_jitter_minutes INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS dim_contract (
    contract_id UUID PRIMARY KEY DEFAULT uuid(),
    venue VARCHAR(50) NOT NULL CHECK (venue IN ('polymarket', 'kalshi', 'other')),
    venue_market_id VARCHAR(100) NOT NULL,
    ticker VARCHAR(100),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    resolution_time TIMESTAMPTZ,
    resolution_rule_text TEXT,
    outcome_type VARCHAR(20) NOT NULL CHECK (outcome_type IN ('binary', 'multi')),
    outcomes JSON,
    status VARCHAR(20) NOT NULL DEFAULT 'draft' CHECK (status IN ('active', 'resolved', 'closed', 'draft')),
    created_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    source_id UUID REFERENCES dim_source(source_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(venue, venue_market_id)
);

-- ============================================================
-- RAW TABLES (append-only, schema-flexible)
-- ============================================================

CREATE TABLE IF NOT EXISTS raw_fred_observations (
    id UUID PRIMARY KEY DEFAULT uuid(),
    series_code VARCHAR(100) NOT NULL,
    observation_date DATE NOT NULL,
    value NUMERIC,
    realtime_start DATE,
    realtime_end DATE,
    raw_data JSON NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id UUID REFERENCES meta_pipeline_runs(run_id),
    UNIQUE(series_code, observation_date, realtime_start)
);

CREATE TABLE IF NOT EXISTS raw_gdelt_events (
    id UUID PRIMARY KEY DEFAULT uuid(),
    gdelt_event_id BIGINT UNIQUE,
    event_date DATE,
    raw_data JSON NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE TABLE IF NOT EXISTS raw_polymarket_markets (
    id UUID PRIMARY KEY DEFAULT uuid(),
    venue_market_id VARCHAR(100) NOT NULL UNIQUE,
    raw_data JSON NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE TABLE IF NOT EXISTS raw_polymarket_prices (
    id UUID PRIMARY KEY DEFAULT uuid(),
    venue_market_id VARCHAR(100) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    outcome VARCHAR(50) NOT NULL DEFAULT 'YES',
    price NUMERIC,
    raw_data JSON NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id UUID REFERENCES meta_pipeline_runs(run_id),
    UNIQUE(venue_market_id, ts, outcome)
);

CREATE TABLE IF NOT EXISTS raw_polymarket_trades (
    id UUID PRIMARY KEY DEFAULT uuid(),
    venue_market_id VARCHAR(100) NOT NULL,
    trade_id VARCHAR(100) UNIQUE,
    ts TIMESTAMPTZ NOT NULL,
    price NUMERIC,
    size NUMERIC,
    side VARCHAR(10),
    raw_data JSON NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE TABLE IF NOT EXISTS raw_prices_ohlcv (
    id UUID PRIMARY KEY DEFAULT uuid(),
    ticker VARCHAR(50) NOT NULL,
    exchange VARCHAR(50),
    date DATE NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    adj_close NUMERIC,
    volume BIGINT,
    split_ratio VARCHAR(20),
    dividend NUMERIC DEFAULT 0,
    raw_data JSON NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id UUID REFERENCES meta_pipeline_runs(run_id),
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS raw_polymarket_orderbook_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid(),
    venue_market_id VARCHAR(100) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    best_bid NUMERIC,
    best_ask NUMERIC,
    spread NUMERIC,
    bids JSON,
    asks JSON,
    raw_data JSON NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id UUID REFERENCES meta_pipeline_runs(run_id),
    UNIQUE(venue_market_id, ts)
);

CREATE TABLE IF NOT EXISTS raw_factor_returns (
    id UUID PRIMARY KEY DEFAULT uuid(),
    date DATE NOT NULL UNIQUE,
    mkt_rf NUMERIC,
    smb NUMERIC,
    hml NUMERIC,
    rmw NUMERIC,
    cma NUMERIC,
    mom NUMERIC,
    rf NUMERIC,
    raw_data JSON NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE TABLE IF NOT EXISTS raw_sec_fundamentals (
    id BIGINT PRIMARY KEY DEFAULT nextval('raw_sec_fundamentals_seq'),
    ticker VARCHAR(20) NOT NULL,
    cik INTEGER NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_label VARCHAR(200),
    metric_value NUMERIC,
    units VARCHAR(50),
    fiscal_period_end DATE NOT NULL,
    filing_date DATE NOT NULL,
    form_type VARCHAR(10),
    accession_number VARCHAR(30),
    fiscal_year INTEGER,
    fiscal_period VARCHAR(10),
    is_amendment BOOLEAN DEFAULT FALSE,
    original_form_type VARCHAR(10),
    filing_sequence INTEGER DEFAULT 1,
    raw_data JSON,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(ticker, metric_name, fiscal_period_end, form_type, accession_number)
);

CREATE TABLE IF NOT EXISTS raw_sec_insider_trades (
    id BIGINT PRIMARY KEY DEFAULT nextval('raw_sec_insider_trades_seq'),
    ticker VARCHAR(20),
    cik INTEGER NOT NULL,
    insider_cik INTEGER,
    insider_name VARCHAR(200),
    insider_title VARCHAR(100),
    transaction_date DATE,
    transaction_type VARCHAR(10),
    shares NUMERIC,
    price_per_share NUMERIC,
    shares_after NUMERIC,
    ownership_type VARCHAR(1),
    form_type VARCHAR(10) DEFAULT '4',
    accession_number VARCHAR(30),
    filing_date DATE NOT NULL,
    raw_data JSON,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(accession_number, insider_cik, transaction_date, transaction_type, shares)
);

CREATE TABLE IF NOT EXISTS raw_sec_13f_holdings (
    id BIGINT PRIMARY KEY DEFAULT nextval('raw_sec_13f_holdings_seq'),
    filer_cik INTEGER NOT NULL,
    filer_name VARCHAR(300),
    report_date DATE NOT NULL,
    filing_date DATE NOT NULL,
    cusip VARCHAR(9),
    issuer_name VARCHAR(300),
    class_title VARCHAR(100),
    market_value BIGINT,
    shares_held BIGINT,
    shares_type VARCHAR(10),
    put_call VARCHAR(10),
    investment_discretion VARCHAR(10),
    voting_authority_sole BIGINT,
    voting_authority_shared BIGINT,
    voting_authority_none BIGINT,
    accession_number VARCHAR(30),
    raw_data JSON,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(accession_number, cusip, filer_cik, report_date)
);

CREATE TABLE IF NOT EXISTS raw_options_chain (
    id BIGINT PRIMARY KEY DEFAULT nextval('raw_options_chain_seq'),
    ticker VARCHAR(20) NOT NULL,
    quote_date DATE NOT NULL,
    expiration DATE NOT NULL,
    strike NUMERIC NOT NULL,
    option_type VARCHAR(4) NOT NULL CHECK (option_type IN ('call', 'put')),
    last_price NUMERIC,
    bid NUMERIC,
    ask NUMERIC,
    volume INTEGER,
    open_interest INTEGER,
    implied_volatility NUMERIC,
    in_the_money BOOLEAN,
    raw_data JSON,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(ticker, quote_date, expiration, strike, option_type)
);

CREATE TABLE IF NOT EXISTS raw_earnings_calendar (
    id BIGINT PRIMARY KEY DEFAULT nextval('raw_earnings_calendar_seq'),
    ticker VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    fiscal_quarter_end DATE,
    eps_estimate NUMERIC,
    eps_actual NUMERIC,
    revenue_estimate NUMERIC,
    revenue_actual NUMERIC,
    report_time VARCHAR(10),
    raw_data JSON,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(ticker, report_date)
);

CREATE TABLE IF NOT EXISTS raw_reddit_posts (
    id BIGINT PRIMARY KEY DEFAULT nextval('raw_reddit_posts_seq'),
    post_id VARCHAR(20) NOT NULL UNIQUE,
    subreddit VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    selftext TEXT,
    author VARCHAR(100),
    score INTEGER,
    upvote_ratio NUMERIC,
    num_comments INTEGER,
    created_utc TIMESTAMPTZ NOT NULL,
    tickers_mentioned JSON,
    sentiment_score NUMERIC,
    raw_data JSON,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS raw_short_interest (
    id BIGINT PRIMARY KEY DEFAULT nextval('raw_short_interest_seq'),
    ticker VARCHAR(20) NOT NULL,
    settlement_date DATE NOT NULL,
    short_interest BIGINT NOT NULL,
    avg_daily_volume BIGINT,
    days_to_cover NUMERIC,
    raw_data JSON,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(ticker, settlement_date)
);

CREATE TABLE IF NOT EXISTS raw_etf_flows (
    id BIGINT PRIMARY KEY DEFAULT nextval('raw_etf_flows_seq'),
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    fund_flow NUMERIC,
    aum NUMERIC,
    shares_outstanding BIGINT,
    raw_data JSON,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS raw_corporate_actions (
    id UUID PRIMARY KEY DEFAULT uuid(),
    ticker VARCHAR(50) NOT NULL,
    action_type VARCHAR(20) NOT NULL CHECK (action_type IN ('split', 'dividend', 'spinoff', 'merger')),
    action_date DATE NOT NULL,
    ratio NUMERIC,
    amount NUMERIC,
    raw_data JSON,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id UUID,
    UNIQUE(ticker, action_type, action_date)
);

CREATE TABLE IF NOT EXISTS raw_ticker_info (
    id UUID PRIMARY KEY DEFAULT uuid(),
    ticker VARCHAR(50) NOT NULL UNIQUE,
    exchange VARCHAR(50),
    asset_class VARCHAR(20) DEFAULT 'equity',
    first_trade_date DATE,
    last_trade_date DATE,
    is_delisted BOOLEAN DEFAULT FALSE,
    delisted_date DATE,
    raw_data JSON,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    run_id UUID
);

-- ============================================================
-- CURATED TABLES (deduplicated, typed, time-correct)
-- ============================================================

CREATE TABLE IF NOT EXISTS cur_prices_ohlcv_daily (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    date DATE NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    adj_close NUMERIC,
    volume BIGINT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    has_adjustment BOOLEAN DEFAULT FALSE,
    adj_ratio NUMERIC DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, date)
);

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
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, date)
);

CREATE TABLE IF NOT EXISTS cur_corporate_actions (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    action_type VARCHAR(20) NOT NULL CHECK (action_type IN ('split', 'dividend', 'spinoff', 'merger')),
    action_date DATE NOT NULL,
    ratio NUMERIC,
    amount NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, action_type, action_date)
);

CREATE TABLE IF NOT EXISTS cur_fundamentals_quarterly (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    fiscal_period_end DATE NOT NULL,
    filing_date DATE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    units VARCHAR(50),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, fiscal_period_end, metric_name)
);

CREATE TABLE IF NOT EXISTS cur_macro_observations (
    series_id UUID NOT NULL REFERENCES dim_macro_series(series_id),
    period_start DATE,
    period_end DATE NOT NULL,
    value NUMERIC,
    revision_id VARCHAR(20),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (series_id, period_end, revision_id)
);

CREATE TABLE IF NOT EXISTS cur_macro_releases (
    release_id UUID PRIMARY KEY DEFAULT uuid(),
    series_id UUID NOT NULL REFERENCES dim_macro_series(series_id),
    release_time TIMESTAMPTZ NOT NULL,
    actual_value NUMERIC,
    forecast_value NUMERIC,
    previous_value NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS cur_news_items (
    news_id UUID PRIMARY KEY DEFAULT uuid(),
    source_id UUID REFERENCES dim_source(source_id),
    headline TEXT NOT NULL,
    body TEXT,
    url VARCHAR(1000),
    language VARCHAR(10),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    entities JSON,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS cur_world_events (
    event_id UUID PRIMARY KEY DEFAULT uuid(),
    source_id UUID REFERENCES dim_source(source_id),
    gdelt_event_id BIGINT,
    event_type VARCHAR(50),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    location JSON,
    actors JSON,
    themes JSON,
    tone_score NUMERIC,
    sentiment_positive NUMERIC,
    sentiment_negative NUMERIC,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS cur_contract_state_daily (
    contract_id UUID NOT NULL REFERENCES dim_contract(contract_id),
    date DATE NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('active', 'resolved', 'closed', 'draft')),
    resolution_time TIMESTAMPTZ,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (contract_id, date)
);

CREATE TABLE IF NOT EXISTS cur_contract_prices (
    contract_id UUID NOT NULL REFERENCES dim_contract(contract_id),
    ts TIMESTAMPTZ NOT NULL,
    outcome VARCHAR(50) NOT NULL,
    price_raw NUMERIC NOT NULL,
    price_normalized NUMERIC NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (contract_id, ts, outcome)
);

CREATE TABLE IF NOT EXISTS cur_contract_orderbook_snapshots (
    contract_id UUID NOT NULL REFERENCES dim_contract(contract_id),
    ts TIMESTAMPTZ NOT NULL,
    best_bid NUMERIC,
    best_ask NUMERIC,
    spread NUMERIC,
    bids JSON,
    asks JSON,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (contract_id, ts)
);

CREATE TABLE IF NOT EXISTS cur_contract_trades (
    contract_id UUID NOT NULL REFERENCES dim_contract(contract_id),
    trade_id VARCHAR(100) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    price NUMERIC NOT NULL,
    size NUMERIC NOT NULL,
    side VARCHAR(10) CHECK (side IN ('buy', 'sell')),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (contract_id, trade_id)
);

CREATE TABLE IF NOT EXISTS cur_contract_resolution (
    contract_id UUID PRIMARY KEY REFERENCES dim_contract(contract_id),
    resolved_time TIMESTAMPTZ NOT NULL,
    resolved_outcome VARCHAR(100) NOT NULL,
    resolution_source_url VARCHAR(1000),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS cur_factor_returns (
    date DATE PRIMARY KEY,
    mkt_rf NUMERIC,
    smb NUMERIC,
    hml NUMERIC,
    rmw NUMERIC,
    cma NUMERIC,
    mom NUMERIC,
    rf NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) NOT NULL DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS cur_insider_trades (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    insider_name VARCHAR(200) NOT NULL,
    insider_title VARCHAR(100),
    transaction_date DATE NOT NULL,
    transaction_type VARCHAR(10) NOT NULL,
    shares NUMERIC NOT NULL,
    price_per_share NUMERIC,
    total_value NUMERIC,
    shares_after NUMERIC,
    ownership_type VARCHAR(1),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'confirmed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, insider_name, transaction_date, transaction_type, shares)
);

CREATE TABLE IF NOT EXISTS cur_institutional_holdings (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    filer_entity_id UUID REFERENCES dim_entity(entity_id),
    filer_name VARCHAR(300) NOT NULL,
    report_date DATE NOT NULL,
    market_value BIGINT,
    shares_held BIGINT NOT NULL,
    shares_type VARCHAR(10),
    put_call VARCHAR(10),
    pct_of_portfolio NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'confirmed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, filer_name, report_date)
);

CREATE TABLE IF NOT EXISTS cur_options_summary_daily (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    date DATE NOT NULL,
    iv_30d NUMERIC,
    iv_60d NUMERIC,
    iv_90d NUMERIC,
    iv_atm_call NUMERIC,
    iv_atm_put NUMERIC,
    put_call_volume_ratio NUMERIC,
    put_call_oi_ratio NUMERIC,
    total_call_volume BIGINT,
    total_put_volume BIGINT,
    total_call_oi BIGINT,
    total_put_oi BIGINT,
    skew_25d NUMERIC,
    iv_term_slope NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, date)
);

CREATE TABLE IF NOT EXISTS cur_earnings_events (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    report_date DATE NOT NULL,
    fiscal_quarter_end DATE,
    eps_estimate NUMERIC,
    eps_actual NUMERIC,
    eps_surprise NUMERIC,
    eps_surprise_pct NUMERIC,
    revenue_estimate NUMERIC,
    revenue_actual NUMERIC,
    revenue_surprise NUMERIC,
    revenue_surprise_pct NUMERIC,
    report_time VARCHAR(10),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'confirmed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, report_date)
);

CREATE TABLE IF NOT EXISTS cur_short_interest (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    settlement_date DATE NOT NULL,
    short_interest BIGINT NOT NULL,
    avg_daily_volume BIGINT,
    days_to_cover NUMERIC,
    short_pct_float NUMERIC,
    short_interest_change NUMERIC,
    short_interest_change_pct NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, settlement_date)
);

CREATE TABLE IF NOT EXISTS cur_etf_flows_daily (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    date DATE NOT NULL,
    fund_flow NUMERIC NOT NULL,
    aum NUMERIC,
    flow_pct_aum NUMERIC,
    flow_5d_sum NUMERIC,
    flow_20d_sum NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, date)
);

-- ============================================================
-- SNAPSHOT TABLES (point-in-time training data)
-- ============================================================

CREATE TABLE IF NOT EXISTS snap_contract_features (
    contract_id UUID NOT NULL REFERENCES dim_contract(contract_id),
    asof_ts TIMESTAMPTZ NOT NULL,
    implied_p_yes NUMERIC,
    spread NUMERIC,
    depth_best_bid NUMERIC,
    depth_best_ask NUMERIC,
    volume_24h NUMERIC,
    trade_count_24h INTEGER,
    price_volatility_24h NUMERIC,
    macro_panel JSON,
    news_counts JSON,
    event_counts_24h INTEGER,
    event_tone_avg NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    price_staleness_hours NUMERIC,
    macro_staleness_days NUMERIC,
    has_price_outliers BOOLEAN DEFAULT FALSE,
    outlier_score NUMERIC,
    data_quality_score NUMERIC DEFAULT 100.0,
    last_price_ts TIMESTAMPTZ,
    micro_trade_imbalance NUMERIC,
    micro_buy_sell_ratio NUMERIC,
    micro_price_impact NUMERIC,
    trade_outlier_pct NUMERIC,
    price_volatility_24h_robust NUMERIC,
    trade_imbalance NUMERIC,
    avg_trade_size NUMERIC,
    trade_size_std NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (contract_id, asof_ts)
);

CREATE TABLE IF NOT EXISTS snap_symbol_features (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    asof_ts TIMESTAMPTZ NOT NULL,
    price_latest NUMERIC,
    price_change_1d NUMERIC,
    price_change_7d NUMERIC,
    volume_avg_20d NUMERIC,
    volatility_20d NUMERIC,
    macro_panel JSON,
    news_counts JSON,
    pe_ratio NUMERIC,
    pb_ratio NUMERIC,
    debt_to_equity NUMERIC,
    roe NUMERIC,
    insider_net_shares_90d NUMERIC,
    insider_buy_count_90d INTEGER,
    institutional_holders_count INTEGER,
    iv_30d NUMERIC,
    put_call_volume_ratio NUMERIC,
    skew_25d NUMERIC,
    days_to_next_earnings INTEGER,
    last_eps_surprise_pct NUMERIC,
    short_interest_ratio NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, asof_ts)
);

CREATE TABLE IF NOT EXISTS snap_universe_membership (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    start_date DATE NOT NULL,
    end_date DATE,
    is_delisted BOOLEAN NOT NULL DEFAULT FALSE,
    available_time TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (symbol_id, start_date)
);

-- ============================================================
-- META / DATA QUALITY TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS meta_data_quality_checks (
    check_id UUID PRIMARY KEY DEFAULT uuid(),
    table_name VARCHAR(100) NOT NULL,
    check_name VARCHAR(100) NOT NULL,
    check_type VARCHAR(50) NOT NULL,
    threshold_value NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(table_name, check_name)
);

CREATE TABLE IF NOT EXISTS meta_data_quality_alerts (
    alert_id UUID PRIMARY KEY DEFAULT uuid(),
    check_id UUID REFERENCES meta_data_quality_checks(check_id),
    table_name VARCHAR(100) NOT NULL,
    check_name VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    message TEXT NOT NULL,
    details JSON,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS meta_data_quality_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid(),
    table_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    sample_size INTEGER,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS meta_universe_composition (
    composition_id UUID PRIMARY KEY DEFAULT uuid(),
    asof_date DATE NOT NULL,
    total_tickers INTEGER NOT NULL,
    active_tickers INTEGER NOT NULL,
    delisted_tickers INTEGER NOT NULL,
    new_listings INTEGER DEFAULT 0,
    delistings INTEGER DEFAULT 0,
    survivor_bias_pct NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS meta_data_staleness (
    staleness_id UUID PRIMARY KEY DEFAULT uuid(),
    table_name VARCHAR(100) NOT NULL,
    latest_timestamp TIMESTAMPTZ,
    staleness_hours NUMERIC,
    threshold_hours NUMERIC,
    is_stale BOOLEAN,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS meta_latency_stats (
    stat_id UUID PRIMARY KEY DEFAULT uuid(),
    source_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(20) NOT NULL,
    metric_value NUMERIC,
    sample_size INTEGER,
    window_start DATE,
    window_end DATE,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    UNIQUE(source_name, metric_name, window_start, window_end)
);

CREATE TABLE IF NOT EXISTS meta_dataset_coverage (
    coverage_id UUID PRIMARY KEY DEFAULT uuid(),
    source_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

-- ============================================================
-- MODEL EVALUATION TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS meta_model_runs (
    run_id UUID PRIMARY KEY,
    model_name VARCHAR(200) NOT NULL,
    scope VARCHAR(50) NOT NULL,
    dataset_id VARCHAR(200),
    config_json JSON,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS meta_model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid(),
    run_id UUID REFERENCES meta_model_runs(run_id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS meta_model_regime_metrics (
    id UUID PRIMARY KEY DEFAULT uuid(),
    run_id UUID REFERENCES meta_model_runs(run_id),
    regime VARCHAR(20) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS meta_factor_exposures (
    id UUID PRIMARY KEY DEFAULT uuid(),
    run_id UUID REFERENCES meta_model_runs(run_id),
    factor VARCHAR(50) NOT NULL,
    beta NUMERIC,
    t_stat NUMERIC,
    p_value NUMERIC,
    r2 NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS meta_stress_results (
    id UUID PRIMARY KEY DEFAULT uuid(),
    run_id UUID REFERENCES meta_model_runs(run_id),
    scenario VARCHAR(50) NOT NULL,
    var NUMERIC,
    es NUMERIC,
    max_dd NUMERIC,
    recovery_days INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp
);

-- ============================================================
-- SEED DATA
-- ============================================================

INSERT INTO meta_data_quality_checks (table_name, check_name, check_type, threshold_value)
VALUES
    ('cur_prices_ohlcv_daily', 'freshness', 'staleness', 48),
    ('cur_contract_prices', 'freshness', 'staleness', 2),
    ('cur_macro_observations', 'freshness', 'staleness', 168),
    ('cur_prices_ohlcv_daily', 'completeness', 'null_rate', 5),
    ('cur_contract_prices', 'completeness', 'null_rate', 1),
    ('dim_symbol', 'survivor_bias', 'delisted_pct', 5)
ON CONFLICT (table_name, check_name) DO NOTHING;

-- ============================================================
-- VIEWS
-- ============================================================

CREATE OR REPLACE VIEW v_data_quality_summary AS
SELECT
    table_name,
    COUNT(*) FILTER (WHERE severity = 'critical' AND resolved_at IS NULL) as critical_alerts,
    COUNT(*) FILTER (WHERE severity = 'error' AND resolved_at IS NULL) as error_alerts,
    COUNT(*) FILTER (WHERE severity = 'warning' AND resolved_at IS NULL) as warning_alerts,
    MAX(created_at) FILTER (WHERE resolved_at IS NULL) as latest_alert_at
FROM meta_data_quality_alerts
GROUP BY table_name;
