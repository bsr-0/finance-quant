-- Curated tables for public markets

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
    time_quality VARCHAR(20) DEFAULT 'assumed' CHECK (time_quality IN ('assumed', 'confirmed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, date)
);

CREATE INDEX idx_cur_prices_available_time ON cur_prices_ohlcv_daily(available_time);
CREATE INDEX idx_cur_prices_event_time ON cur_prices_ohlcv_daily(event_time);

CREATE TABLE IF NOT EXISTS cur_corporate_actions (
    symbol_id UUID NOT NULL REFERENCES dim_symbol(symbol_id),
    action_type VARCHAR(20) NOT NULL CHECK (action_type IN ('split', 'dividend', 'spinoff', 'merger')),
    action_date DATE NOT NULL,
    ratio NUMERIC,
    amount NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, action_type, action_date)
);

CREATE INDEX idx_cur_corp_actions_available_time ON cur_corporate_actions(available_time);

CREATE TABLE IF NOT EXISTS cur_fundamentals_quarterly (
    entity_id UUID NOT NULL REFERENCES dim_entity(entity_id),
    fiscal_period_end DATE NOT NULL,
    filing_date DATE,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    units VARCHAR(50),
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (entity_id, fiscal_period_end, metric_name)
);

CREATE INDEX idx_cur_fundamentals_available_time ON cur_fundamentals_quarterly(available_time);
