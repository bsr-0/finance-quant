-- Model evaluation metadata tables

CREATE TABLE IF NOT EXISTS meta_model_runs (
    run_id UUID PRIMARY KEY,
    model_name VARCHAR(200) NOT NULL,
    scope VARCHAR(50) NOT NULL,
    dataset_id VARCHAR(200),
    config_json JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS meta_model_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES meta_model_runs(run_id),
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS meta_model_regime_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES meta_model_runs(run_id),
    regime VARCHAR(20) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS meta_factor_exposures (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES meta_model_runs(run_id),
    factor VARCHAR(50) NOT NULL,
    beta NUMERIC,
    t_stat NUMERIC,
    p_value NUMERIC,
    r2 NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS meta_stress_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES meta_model_runs(run_id),
    scenario VARCHAR(50) NOT NULL,
    var NUMERIC,
    es NUMERIC,
    max_dd NUMERIC,
    recovery_days INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
