-- Raw and curated tables for options / implied volatility data

CREATE TABLE IF NOT EXISTS raw_options_chain (
    id BIGSERIAL PRIMARY KEY,
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
    raw_data JSONB,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(ticker, quote_date, expiration, strike, option_type)
);

CREATE INDEX idx_raw_options_ticker ON raw_options_chain(ticker);
CREATE INDEX idx_raw_options_quote ON raw_options_chain(quote_date);
CREATE INDEX idx_raw_options_expiry ON raw_options_chain(expiration);

-- Daily aggregated options summary per symbol
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
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, date)
);

CREATE INDEX idx_cur_options_summary_available_time ON cur_options_summary_daily(available_time);
