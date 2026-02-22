-- Meta tables for pipeline tracking and audit

CREATE TABLE IF NOT EXISTS meta_pipeline_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    git_sha VARCHAR(40),
    pipeline_name VARCHAR(100) NOT NULL,
    params JSONB,
    status VARCHAR(20) NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    row_counts JSONB,
    errors TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_meta_pipeline_runs_status ON meta_pipeline_runs(status);
CREATE INDEX idx_meta_pipeline_runs_started_at ON meta_pipeline_runs(started_at);
CREATE INDEX idx_meta_pipeline_runs_pipeline_name ON meta_pipeline_runs(pipeline_name);
