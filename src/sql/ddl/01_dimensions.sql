-- Dimension tables (common and market-specific)

-- Common dimensions
CREATE TABLE IF NOT EXISTS dim_source (
    source_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) NOT NULL UNIQUE,
    type VARCHAR(20) NOT NULL CHECK (type IN ('api', 'files', 'scrape')),
    base_url VARCHAR(500),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_calendar_market (
    cal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(50) NOT NULL UNIQUE,
    timezone VARCHAR(50) NOT NULL DEFAULT 'UTC',
    trading_days TEXT, -- representation or link to generated table
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dim_entity (
    entity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type VARCHAR(30) NOT NULL CHECK (entity_type IN ('company', 'country', 'person', 'indicator', 'topic', 'organization')),
    name VARCHAR(200) NOT NULL,
    aliases JSONB,
    external_ids JSONB, -- cik, figi, isin, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_entity_type ON dim_entity(entity_type);
CREATE INDEX IF NOT EXISTS idx_dim_entity_name ON dim_entity(name);

-- Market dimensions
CREATE TABLE IF NOT EXISTS dim_symbol (
    symbol_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(50) NOT NULL,
    exchange VARCHAR(50),
    asset_class VARCHAR(20) NOT NULL CHECK (asset_class IN ('equity', 'etf', 'index', 'fx', 'commodity', 'crypto')),
    currency VARCHAR(10) NOT NULL DEFAULT 'USD',
    start_date DATE,
    end_date DATE,
    is_delisted BOOLEAN NOT NULL DEFAULT FALSE,
    external_ids JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(ticker, exchange)
);

CREATE INDEX IF NOT EXISTS idx_dim_symbol_ticker ON dim_symbol(ticker);
CREATE INDEX IF NOT EXISTS idx_dim_symbol_asset_class ON dim_symbol(asset_class);
CREATE INDEX IF NOT EXISTS idx_dim_symbol_is_delisted ON dim_symbol(is_delisted);

-- Macro dimensions
CREATE TABLE IF NOT EXISTS dim_macro_series (
    series_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_series_code VARCHAR(100) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    units VARCHAR(50),
    frequency VARCHAR(20) CHECK (frequency IN ('daily', 'weekly', 'monthly', 'quarterly', 'annual')),
    country VARCHAR(10),
    source_id UUID REFERENCES dim_source(source_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dim_macro_series_country ON dim_macro_series(country);

-- Prediction market dimensions
CREATE TABLE IF NOT EXISTS dim_contract (
    contract_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venue VARCHAR(50) NOT NULL CHECK (venue IN ('polymarket', 'kalshi', 'other')),
    venue_market_id VARCHAR(100) NOT NULL,
    ticker VARCHAR(100),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    resolution_time TIMESTAMPTZ,
    resolution_rule_text TEXT,
    outcome_type VARCHAR(20) NOT NULL CHECK (outcome_type IN ('binary', 'multi')),
    outcomes JSONB,
    status VARCHAR(20) NOT NULL DEFAULT 'draft' CHECK (status IN ('active', 'resolved', 'closed', 'draft')),
    created_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    source_id UUID REFERENCES dim_source(source_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(venue, venue_market_id)
);

CREATE INDEX IF NOT EXISTS idx_dim_contract_venue ON dim_contract(venue);
CREATE INDEX IF NOT EXISTS idx_dim_contract_status ON dim_contract(status);
CREATE INDEX IF NOT EXISTS idx_dim_contract_resolution_time ON dim_contract(resolution_time);
CREATE INDEX IF NOT EXISTS idx_dim_contract_available_time ON dim_contract(available_time);
