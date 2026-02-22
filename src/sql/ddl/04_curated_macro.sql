-- Curated tables for macro and economic data

CREATE TABLE IF NOT EXISTS cur_macro_observations (
    series_id UUID NOT NULL REFERENCES dim_macro_series(series_id),
    period_start DATE,
    period_end DATE NOT NULL,
    value NUMERIC,
    revision_id INTEGER, -- NULL for initial release
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (series_id, period_end, revision_id)
);

CREATE INDEX idx_cur_macro_available_time ON cur_macro_observations(available_time);
CREATE INDEX idx_cur_macro_period_end ON cur_macro_observations(period_end);

CREATE TABLE IF NOT EXISTS cur_macro_releases (
    release_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    series_id UUID NOT NULL REFERENCES dim_macro_series(series_id),
    release_time TIMESTAMPTZ NOT NULL,
    actual_value NUMERIC,
    forecast_value NUMERIC,
    previous_value NUMERIC,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_cur_macro_releases_series_time ON cur_macro_releases(series_id, release_time);
CREATE INDEX idx_cur_macro_releases_available_time ON cur_macro_releases(available_time);
