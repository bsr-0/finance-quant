-- Curated tables for insider trading and institutional holdings

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
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, insider_name, transaction_date, transaction_type, shares)
);

CREATE INDEX idx_cur_insider_available_time ON cur_insider_trades(available_time);
CREATE INDEX idx_cur_insider_event_time ON cur_insider_trades(event_time);
CREATE INDEX idx_cur_insider_txn_type ON cur_insider_trades(transaction_type);

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
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, filer_name, report_date)
);

CREATE INDEX idx_cur_inst_holdings_available_time ON cur_institutional_holdings(available_time);
CREATE INDEX idx_cur_inst_holdings_report_date ON cur_institutional_holdings(report_date);
CREATE INDEX idx_cur_inst_holdings_filer ON cur_institutional_holdings(filer_name);
