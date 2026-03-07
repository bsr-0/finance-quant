-- Raw and curated tables for earnings calendar and estimates

CREATE TABLE IF NOT EXISTS raw_earnings_calendar (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    fiscal_quarter_end DATE,
    eps_estimate NUMERIC,
    eps_actual NUMERIC,
    revenue_estimate NUMERIC,
    revenue_actual NUMERIC,
    report_time VARCHAR(10),
    raw_data JSONB,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(ticker, report_date)
);

CREATE INDEX IF NOT EXISTS idx_raw_earnings_ticker ON raw_earnings_calendar(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_earnings_date ON raw_earnings_calendar(report_date);

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
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, report_date)
);

CREATE INDEX IF NOT EXISTS idx_cur_earnings_available_time ON cur_earnings_events(available_time);
CREATE INDEX IF NOT EXISTS idx_cur_earnings_event_time ON cur_earnings_events(event_time);
