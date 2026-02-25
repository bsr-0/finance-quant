-- Curated tables for prediction markets

CREATE TABLE IF NOT EXISTS cur_contract_state_daily (
    contract_id UUID NOT NULL REFERENCES dim_contract(contract_id),
    date DATE NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('active', 'resolved', 'closed', 'draft')),
    resolution_time TIMESTAMPTZ,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (contract_id, date)
);

CREATE INDEX idx_cur_contract_state_available_time ON cur_contract_state_daily(available_time);

CREATE TABLE IF NOT EXISTS cur_contract_prices (
    contract_id UUID NOT NULL REFERENCES dim_contract(contract_id),
    ts TIMESTAMPTZ NOT NULL,
    outcome VARCHAR(50) NOT NULL,
    price_raw NUMERIC NOT NULL,
    price_normalized NUMERIC NOT NULL, -- 0..1 for binary
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (contract_id, ts, outcome)
);

CREATE INDEX idx_cur_contract_prices_contract_ts ON cur_contract_prices(contract_id, ts);
CREATE INDEX idx_cur_contract_prices_available_time ON cur_contract_prices(available_time);

CREATE TABLE IF NOT EXISTS cur_contract_orderbook_snapshots (
    contract_id UUID NOT NULL REFERENCES dim_contract(contract_id),
    ts TIMESTAMPTZ NOT NULL,
    best_bid NUMERIC,
    best_ask NUMERIC,
    spread NUMERIC,
    bids JSONB, -- array of [price, size]
    asks JSONB, -- array of [price, size]
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (contract_id, ts)
);

CREATE INDEX idx_cur_contract_ob_available_time ON cur_contract_orderbook_snapshots(available_time);
CREATE INDEX idx_cur_contract_ob_contract_ts ON cur_contract_orderbook_snapshots(contract_id, ts);

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
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (contract_id, trade_id)
);

CREATE INDEX idx_cur_contract_trades_contract_ts ON cur_contract_trades(contract_id, ts);
CREATE INDEX idx_cur_contract_trades_available_time ON cur_contract_trades(available_time);

CREATE TABLE IF NOT EXISTS cur_contract_resolution (
    contract_id UUID PRIMARY KEY REFERENCES dim_contract(contract_id),
    resolved_time TIMESTAMPTZ NOT NULL,
    resolved_outcome VARCHAR(100) NOT NULL,
    resolution_source_url VARCHAR(1000),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cur_contract_resolution_available_time ON cur_contract_resolution(available_time);
