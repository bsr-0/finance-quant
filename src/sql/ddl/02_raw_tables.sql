-- Raw tables (append-only, schema-flexible for ingested data)

CREATE TABLE IF NOT EXISTS raw_fred_observations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    series_code VARCHAR(100) NOT NULL,
    observation_date DATE NOT NULL,
    value NUMERIC,
    raw_data JSONB NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE INDEX idx_raw_fred_series_date ON raw_fred_observations(series_code, observation_date);
CREATE INDEX idx_raw_fred_extracted_at ON raw_fred_observations(extracted_at);

CREATE TABLE IF NOT EXISTS raw_gdelt_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gdelt_event_id BIGINT,
    event_date DATE,
    raw_data JSONB NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE INDEX idx_raw_gdelt_event_date ON raw_gdelt_events(event_date);
CREATE INDEX idx_raw_gdelt_extracted_at ON raw_gdelt_events(extracted_at);

CREATE TABLE IF NOT EXISTS raw_polymarket_markets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venue_market_id VARCHAR(100) NOT NULL,
    raw_data JSONB NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE INDEX idx_raw_polymarket_market_id ON raw_polymarket_markets(venue_market_id);
CREATE INDEX idx_raw_polymarket_extracted_at ON raw_polymarket_markets(extracted_at);

CREATE TABLE IF NOT EXISTS raw_polymarket_prices (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venue_market_id VARCHAR(100) NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    outcome VARCHAR(50),
    price NUMERIC,
    raw_data JSONB NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE INDEX idx_raw_polymarket_prices_market_ts ON raw_polymarket_prices(venue_market_id, ts);

CREATE TABLE IF NOT EXISTS raw_polymarket_trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    venue_market_id VARCHAR(100) NOT NULL,
    trade_id VARCHAR(100),
    ts TIMESTAMPTZ NOT NULL,
    price NUMERIC,
    size NUMERIC,
    side VARCHAR(10),
    raw_data JSONB NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE INDEX idx_raw_polymarket_trades_market_ts ON raw_polymarket_trades(venue_market_id, ts);

CREATE TABLE IF NOT EXISTS raw_prices_ohlcv (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(50) NOT NULL,
    exchange VARCHAR(50),
    date DATE NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    adj_close NUMERIC,
    volume BIGINT,
    raw_data JSONB NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id UUID REFERENCES meta_pipeline_runs(run_id)
);

CREATE INDEX idx_raw_prices_ticker_date ON raw_prices_ohlcv(ticker, date);
CREATE INDEX idx_raw_prices_extracted_at ON raw_prices_ohlcv(extracted_at);
