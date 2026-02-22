-- Data quality and monitoring tables

-- Add data quality columns to existing tables
ALTER TABLE cur_prices_ohlcv_daily 
    ADD COLUMN IF NOT EXISTS data_quality_flag VARCHAR(20) DEFAULT 'ok',
    ADD COLUMN IF NOT EXISTS has_adjustment BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS adj_ratio NUMERIC DEFAULT 1.0;

CREATE INDEX IF NOT EXISTS idx_cur_prices_quality 
    ON cur_prices_ohlcv_daily(data_quality_flag) 
    WHERE data_quality_flag != 'ok';

-- Add staleness tracking to snapshots
ALTER TABLE snap_contract_features
    ADD COLUMN IF NOT EXISTS price_staleness_hours NUMERIC,
    ADD COLUMN IF NOT EXISTS macro_staleness_days NUMERIC,
    ADD COLUMN IF NOT EXISTS has_price_outliers BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS outlier_score NUMERIC,
    ADD COLUMN IF NOT EXISTS data_quality_score NUMERIC DEFAULT 100.0,
    ADD COLUMN IF NOT EXISTS last_price_ts TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS micro_trade_imbalance NUMERIC,
    ADD COLUMN IF NOT EXISTS micro_buy_sell_ratio NUMERIC,
    ADD COLUMN IF NOT EXISTS micro_price_impact NUMERIC;

CREATE INDEX IF NOT EXISTS idx_snap_quality 
    ON snap_contract_features(data_quality_score) 
    WHERE data_quality_score < 70;

-- Data quality monitoring tables
CREATE TABLE IF NOT EXISTS meta_data_quality_checks (
    check_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100) NOT NULL,
    check_name VARCHAR(100) NOT NULL,
    check_type VARCHAR(50) NOT NULL,
    threshold_value NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(table_name, check_name)
);

CREATE TABLE IF NOT EXISTS meta_data_quality_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    check_id UUID REFERENCES meta_data_quality_checks(check_id),
    table_name VARCHAR(100) NOT NULL,
    check_name VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    message TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100)
);

CREATE INDEX IF NOT EXISTS idx_dq_alerts_unresolved 
    ON meta_data_quality_alerts(created_at DESC) 
    WHERE resolved_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_dq_alerts_severity 
    ON meta_data_quality_alerts(severity, created_at DESC);

CREATE TABLE IF NOT EXISTS meta_data_quality_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    sample_size INTEGER,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dq_metrics_table 
    ON meta_data_quality_metrics(table_name, metric_name, recorded_at DESC);

-- Universe composition tracking (for survivor bias monitoring)
CREATE TABLE IF NOT EXISTS meta_universe_composition (
    composition_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    asof_date DATE NOT NULL,
    total_tickers INTEGER NOT NULL,
    active_tickers INTEGER NOT NULL,
    delisted_tickers INTEGER NOT NULL,
    new_listings INTEGER DEFAULT 0,
    delistings INTEGER DEFAULT 0,
    survivor_bias_pct NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_universe_composition_date 
    ON meta_universe_composition(asof_date DESC);

-- Corporate actions tracking (enhanced)
CREATE TABLE IF NOT EXISTS raw_corporate_actions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(50) NOT NULL,
    action_type VARCHAR(20) NOT NULL CHECK (action_type IN ('split', 'dividend', 'spinoff', 'merger')),
    action_date DATE NOT NULL,
    ratio NUMERIC,
    amount NUMERIC,
    raw_data JSONB,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id UUID,
    UNIQUE(ticker, action_type, action_date)
);

CREATE INDEX IF NOT EXISTS idx_raw_corp_actions_ticker 
    ON raw_corporate_actions(ticker, action_date);

-- Ticker info tracking (for delisting detection)
CREATE TABLE IF NOT EXISTS raw_ticker_info (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(50) NOT NULL UNIQUE,
    exchange VARCHAR(50),
    asset_class VARCHAR(20) DEFAULT 'equity',
    first_trade_date DATE,
    last_trade_date DATE,
    is_delisted BOOLEAN DEFAULT FALSE,
    delisted_date DATE,
    raw_data JSONB,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id UUID
);

CREATE INDEX IF NOT EXISTS idx_raw_ticker_delisted 
    ON raw_ticker_info(is_delisted) 
    WHERE is_delisted = TRUE;

-- Data staleness tracking
CREATE TABLE IF NOT EXISTS meta_data_staleness (
    staleness_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    table_name VARCHAR(100) NOT NULL,
    latest_timestamp TIMESTAMPTZ,
    staleness_hours NUMERIC,
    threshold_hours NUMERIC,
    is_stale BOOLEAN,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_staleness_table 
    ON meta_data_staleness(table_name, checked_at DESC);

-- Insert default data quality checks
INSERT INTO meta_data_quality_checks (table_name, check_name, check_type, threshold_value)
VALUES 
    ('cur_prices_ohlcv_daily', 'freshness', 'staleness', 48),
    ('cur_contract_prices', 'freshness', 'staleness', 2),
    ('cur_macro_observations', 'freshness', 'staleness', 168),
    ('cur_prices_ohlcv_daily', 'completeness', 'null_rate', 5),
    ('cur_contract_prices', 'completeness', 'null_rate', 1),
    ('dim_symbol', 'survivor_bias', 'delisted_pct', 5)
ON CONFLICT (table_name, check_name) DO NOTHING;

-- Function to record staleness check
CREATE OR REPLACE FUNCTION record_staleness_check(
    p_table_name TEXT,
    p_timestamp_col TEXT DEFAULT 'available_time',
    p_threshold_hours NUMERIC DEFAULT 24
) RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_latest_ts TIMESTAMPTZ;
    v_staleness_hours NUMERIC;
BEGIN
    EXECUTE format(
        'SELECT MAX(%I) FROM %I',
        p_timestamp_col,
        p_table_name
    ) INTO v_latest_ts;
    
    IF v_latest_ts IS NOT NULL THEN
        v_staleness_hours := EXTRACT(EPOCH FROM (NOW() - v_latest_ts)) / 3600;
        
        INSERT INTO meta_data_staleness 
            (table_name, latest_timestamp, staleness_hours, threshold_hours, is_stale)
        VALUES 
            (p_table_name, v_latest_ts, v_staleness_hours, p_threshold_hours, 
             v_staleness_hours > p_threshold_hours);
    END IF;
END;
$$;

-- Function to calculate survivor bias metrics
CREATE OR REPLACE FUNCTION calculate_survivor_bias(
    p_asof_date DATE DEFAULT CURRENT_DATE
) RETURNS TABLE(
    total_tickers INTEGER,
    active_tickers INTEGER,
    delisted_tickers INTEGER,
    survivor_bias_pct NUMERIC
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as total_tickers,
        COUNT(*) FILTER (WHERE NOT is_delisted)::INTEGER as active_tickers,
        COUNT(*) FILTER (WHERE is_delisted)::INTEGER as delisted_tickers,
        ROUND(
            COUNT(*) FILTER (WHERE is_delisted)::NUMERIC / 
            NULLIF(COUNT(*), 0) * 100, 
            2
        ) as survivor_bias_pct
    FROM dim_symbol
    WHERE start_date <= p_asof_date
      AND (end_date IS NULL OR end_date >= p_asof_date);
END;
$$;

-- View for data quality dashboard
CREATE OR REPLACE VIEW v_data_quality_summary AS
SELECT 
    table_name,
    COUNT(*) FILTER (WHERE severity = 'critical' AND resolved_at IS NULL) as critical_alerts,
    COUNT(*) FILTER (WHERE severity = 'error' AND resolved_at IS NULL) as error_alerts,
    COUNT(*) FILTER (WHERE severity = 'warning' AND resolved_at IS NULL) as warning_alerts,
    MAX(created_at) FILTER (WHERE resolved_at IS NULL) as latest_alert_at
FROM meta_data_quality_alerts
GROUP BY table_name;

-- View for staleness dashboard
CREATE OR REPLACE VIEW v_data_staleness_summary AS
SELECT 
    table_name,
    latest_timestamp,
    staleness_hours,
    threshold_hours,
    is_stale,
    CASE 
        WHEN staleness_hours > threshold_hours * 2 THEN 'CRITICAL'
        WHEN staleness_hours > threshold_hours THEN 'WARNING'
        ELSE 'OK'
    END as status
FROM meta_data_staleness
WHERE checked_at = (
    SELECT MAX(checked_at) 
    FROM meta_data_staleness s2 
    WHERE s2.table_name = meta_data_staleness.table_name
);
