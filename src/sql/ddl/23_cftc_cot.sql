-- Raw and curated tables for CFTC Commitments of Traders (COT) data

CREATE TABLE IF NOT EXISTS raw_cftc_cot (
    id BIGSERIAL PRIMARY KEY,
    commodity_code VARCHAR(20) NOT NULL,
    commodity_name VARCHAR(200),
    report_date DATE NOT NULL,
    commercial_long BIGINT,
    commercial_short BIGINT,
    noncommercial_long BIGINT,
    noncommercial_short BIGINT,
    noncommercial_spreading BIGINT,
    nonreportable_long BIGINT,
    nonreportable_short BIGINT,
    open_interest BIGINT,
    raw_data JSONB,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(commodity_code, report_date)
);

CREATE INDEX IF NOT EXISTS idx_raw_cftc_cot_code ON raw_cftc_cot(commodity_code);
CREATE INDEX IF NOT EXISTS idx_raw_cftc_cot_date ON raw_cftc_cot(report_date);

CREATE TABLE IF NOT EXISTS cur_cftc_cot (
    commodity_code VARCHAR(20) NOT NULL,
    report_date DATE NOT NULL,
    commercial_net BIGINT,
    noncommercial_net BIGINT,
    nonreportable_net BIGINT,
    open_interest BIGINT,
    commercial_pct_oi NUMERIC,
    noncommercial_pct_oi NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'confirmed' CHECK (time_quality IN ('assumed', 'confirmed', 'inferred')),
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_quality_flag VARCHAR(50),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (commodity_code, report_date)
);

CREATE INDEX IF NOT EXISTS idx_cur_cftc_cot_available_time ON cur_cftc_cot(available_time);
