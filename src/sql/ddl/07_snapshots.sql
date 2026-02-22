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

CREATE INDEX idx_snap_contract_features_asof_ts ON snap_contract_features(asof_ts);
CREATE INDEX idx_snap_contract_features_contract ON snap_contract_features(contract_id);

-- Optional: Symbol-level snapshots for equity training
CREATE TABLE IF NOT EXISTS snap_symbol_features (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    asof_ts TIMESTAMPTZ NOT NULL,
    price_latest NUMERIC,
    price_change_1d NUMERIC,
    price_change_7d NUMERIC,
    volume_avg_20d NUMERIC,
    volatility_20d NUMERIC,
    macro_panel JSONB,
    news_counts JSONB,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, asof_ts)
);

CREATE INDEX idx_snap_symbol_features_asof_ts ON snap_symbol_features(asof_ts);
