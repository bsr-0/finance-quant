-- Raw and curated tables for short interest data

CREATE TABLE IF NOT EXISTS raw_short_interest (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    settlement_date DATE NOT NULL,
    short_interest BIGINT NOT NULL,
    avg_daily_volume BIGINT,
    days_to_cover NUMERIC,
    raw_data JSONB,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(ticker, settlement_date)
);

CREATE INDEX idx_raw_short_interest_ticker ON raw_short_interest(ticker);
CREATE INDEX idx_raw_short_interest_date ON raw_short_interest(settlement_date);

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
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, settlement_date)
);

CREATE INDEX idx_cur_short_interest_available_time ON cur_short_interest(available_time);
