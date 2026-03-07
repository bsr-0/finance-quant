-- Snapshot tables for training data

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
    macro_panel JSONB, -- key-value of macro series values at asof_ts
    news_counts JSONB, -- counts by time window (1h/24h/7d)
    event_counts_24h INTEGER,
    event_tone_avg NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (contract_id, asof_ts)
);

CREATE INDEX IF NOT EXISTS idx_snap_contract_features_asof_ts ON snap_contract_features(asof_ts);
CREATE INDEX IF NOT EXISTS idx_snap_contract_features_contract ON snap_contract_features(contract_id);

-- Symbol-level snapshots for equity training
CREATE TABLE IF NOT EXISTS snap_symbol_features (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    asof_ts TIMESTAMPTZ NOT NULL,
    -- Price & volume
    price_latest NUMERIC,
    price_change_1d NUMERIC,
    price_change_7d NUMERIC,
    volume_avg_20d NUMERIC,
    volatility_20d NUMERIC,
    -- Macro & news
    macro_panel JSONB,
    news_counts JSONB,
    -- Fundamentals (latest quarterly as of asof_ts)
    pe_ratio NUMERIC,
    pb_ratio NUMERIC,
    debt_to_equity NUMERIC,
    roe NUMERIC,
    -- Insider activity (rolling windows)
    insider_net_shares_90d NUMERIC,
    insider_buy_count_90d INTEGER,
    -- Institutional holdings
    institutional_holders_count INTEGER,
    -- Options / IV
    iv_30d NUMERIC,
    put_call_volume_ratio NUMERIC,
    skew_25d NUMERIC,
    -- Earnings
    days_to_next_earnings INTEGER,
    last_eps_surprise_pct NUMERIC,
    -- Short interest
    short_interest_ratio NUMERIC,
    -- Timestamps
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, asof_ts)
);

CREATE INDEX IF NOT EXISTS idx_snap_symbol_features_asof_ts ON snap_symbol_features(asof_ts);
