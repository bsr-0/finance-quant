-- Factor returns tables

CREATE TABLE IF NOT EXISTS raw_factor_returns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    date DATE NOT NULL,
    mkt_rf NUMERIC,
    smb NUMERIC,
    hml NUMERIC,
    rmw NUMERIC,
    cma NUMERIC,
    mom NUMERIC,
    rf NUMERIC,
    raw_data JSONB NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_factor_returns_date ON raw_factor_returns(date);

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
    time_quality VARCHAR(20) NOT NULL DEFAULT 'assumed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_cur_factor_returns_date ON cur_factor_returns(date);
