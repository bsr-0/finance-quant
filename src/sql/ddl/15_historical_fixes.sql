-- Historical fixes: latency stats + conservative feature columns

CREATE TABLE IF NOT EXISTS meta_latency_stats (
    stat_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(20) NOT NULL,
    metric_value NUMERIC,
    sample_size INTEGER,
    window_start DATE,
    window_end DATE,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_meta_latency_stats_unique
    ON meta_latency_stats (source_name, metric_name, window_start, window_end);

CREATE INDEX IF NOT EXISTS idx_meta_latency_stats_recent
    ON meta_latency_stats (source_name, computed_at DESC);

CREATE TABLE IF NOT EXISTS meta_dataset_coverage (
    coverage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE dim_macro_series
    ADD COLUMN IF NOT EXISTS release_time TIME,
    ADD COLUMN IF NOT EXISTS release_timezone VARCHAR(50),
    ADD COLUMN IF NOT EXISTS release_day_offset INTEGER,
    ADD COLUMN IF NOT EXISTS release_jitter_minutes INTEGER;

ALTER TABLE snap_contract_features
    ADD COLUMN IF NOT EXISTS price_staleness_hours NUMERIC,
    ADD COLUMN IF NOT EXISTS macro_staleness_days NUMERIC,
    ADD COLUMN IF NOT EXISTS trade_outlier_pct NUMERIC,
    ADD COLUMN IF NOT EXISTS price_volatility_24h_robust NUMERIC,
    ADD COLUMN IF NOT EXISTS trade_imbalance NUMERIC,
    ADD COLUMN IF NOT EXISTS avg_trade_size NUMERIC,
    ADD COLUMN IF NOT EXISTS trade_size_std NUMERIC,
    ADD COLUMN IF NOT EXISTS data_quality_score NUMERIC;

CREATE INDEX IF NOT EXISTS idx_snap_contract_quality_score
    ON snap_contract_features (data_quality_score);
