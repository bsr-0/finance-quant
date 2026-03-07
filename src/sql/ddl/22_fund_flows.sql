-- Raw and curated tables for ETF fund flows

CREATE TABLE IF NOT EXISTS raw_etf_flows (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    fund_flow NUMERIC,
    aum NUMERIC,
    shares_outstanding BIGINT,
    raw_data JSONB,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_raw_etf_flows_ticker ON raw_etf_flows(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_etf_flows_date ON raw_etf_flows(date);

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
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, date)
);

CREATE INDEX IF NOT EXISTS idx_cur_etf_flows_available_time ON cur_etf_flows_daily(available_time);
