-- Migration: Add corporate action columns to raw_prices_ohlcv
-- and add realtime columns to raw_fred_observations

ALTER TABLE raw_prices_ohlcv ADD COLUMN IF NOT EXISTS split_ratio VARCHAR(20);
ALTER TABLE raw_prices_ohlcv ADD COLUMN IF NOT EXISTS dividend NUMERIC DEFAULT 0;

ALTER TABLE raw_fred_observations ADD COLUMN IF NOT EXISTS realtime_start DATE;
ALTER TABLE raw_fred_observations ADD COLUMN IF NOT EXISTS realtime_end DATE;
