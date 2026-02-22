-- Time-series partitioning for large tables

-- Enable pg_partman extension if available (for advanced partitioning)
-- CREATE EXTENSION IF NOT EXISTS pg_partman;

-- Partitioned macro observations table
CREATE TABLE IF NOT EXISTS cur_macro_observations_part (
    series_id UUID NOT NULL,
    period_start DATE,
    period_end DATE NOT NULL,
    value NUMERIC,
    revision_id INTEGER,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (series_id, period_end, revision_id)
) PARTITION BY RANGE (period_end);

-- Create partitions for recent years
CREATE TABLE IF NOT EXISTS cur_macro_observations_2020 
    PARTITION OF cur_macro_observations_part
    FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');

CREATE TABLE IF NOT EXISTS cur_macro_observations_2021 
    PARTITION OF cur_macro_observations_part
    FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');

CREATE TABLE IF NOT EXISTS cur_macro_observations_2022 
    PARTITION OF cur_macro_observations_part
    FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');

CREATE TABLE IF NOT EXISTS cur_macro_observations_2023 
    PARTITION OF cur_macro_observations_part
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE IF NOT EXISTS cur_macro_observations_2024 
    PARTITION OF cur_macro_observations_part
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS cur_macro_observations_2025 
    PARTITION OF cur_macro_observations_part
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Default partition for any other dates
CREATE TABLE IF NOT EXISTS cur_macro_observations_default 
    PARTITION OF cur_macro_observations_part DEFAULT;

-- Partitioned prices table
CREATE TABLE IF NOT EXISTS cur_prices_ohlcv_daily_part (
    symbol_id UUID NOT NULL,
    date DATE NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    adj_close NUMERIC,
    volume BIGINT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol_id, date)
) PARTITION BY RANGE (date);

-- Create yearly partitions
CREATE TABLE IF NOT EXISTS cur_prices_2020 
    PARTITION OF cur_prices_ohlcv_daily_part
    FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');

CREATE TABLE IF NOT EXISTS cur_prices_2021 
    PARTITION OF cur_prices_ohlcv_daily_part
    FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');

CREATE TABLE IF NOT EXISTS cur_prices_2022 
    PARTITION OF cur_prices_ohlcv_daily_part
    FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');

CREATE TABLE IF NOT EXISTS cur_prices_2023 
    PARTITION OF cur_prices_ohlcv_daily_part
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE IF NOT EXISTS cur_prices_2024 
    PARTITION OF cur_prices_ohlcv_daily_part
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS cur_prices_2025 
    PARTITION OF cur_prices_ohlcv_daily_part
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS cur_prices_default 
    PARTITION OF cur_prices_ohlcv_daily_part DEFAULT;

-- Partitioned contract prices (high frequency data)
CREATE TABLE IF NOT EXISTS cur_contract_prices_part (
    contract_id UUID NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    outcome VARCHAR(50) NOT NULL,
    price_raw NUMERIC NOT NULL,
    price_normalized NUMERIC NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    available_time TIMESTAMPTZ NOT NULL,
    time_quality VARCHAR(20) DEFAULT 'assumed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (contract_id, ts, outcome)
) PARTITION BY RANGE (ts);

-- Create monthly partitions for contract prices
CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_01 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_02 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_03 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-03-01') TO ('2024-04-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_04 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-04-01') TO ('2024-05-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_05 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-05-01') TO ('2024-06-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_06 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-06-01') TO ('2024-07-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_07 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-07-01') TO ('2024-08-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_08 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-08-01') TO ('2024-09-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_09 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-09-01') TO ('2024-10-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_10 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-10-01') TO ('2024-11-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_11 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-11-01') TO ('2024-12-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_2024_12 
    PARTITION OF cur_contract_prices_part
    FOR VALUES FROM ('2024-12-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS cur_contract_prices_default 
    PARTITION OF cur_contract_prices_part DEFAULT;

-- Function to create new partitions automatically
CREATE OR REPLACE FUNCTION create_monthly_partition(
    p_table_name TEXT,
    p_year INT,
    p_month INT
) RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    partition_name := p_table_name || '_' || p_year || '_' || LPAD(p_month::TEXT, 2, '0');
    start_date := MAKE_DATE(p_year, p_month, 1);
    end_date := start_date + INTERVAL '1 month';
    
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
        partition_name,
        p_table_name,
        start_date,
        end_date
    );
    
    RETURN partition_name;
END;
$$;

-- Function to create yearly partitions
CREATE OR REPLACE FUNCTION create_yearly_partition(
    p_table_name TEXT,
    p_year INT
) RETURNS TEXT
LANGUAGE plpgsql
AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    partition_name := p_table_name || '_' || p_year;
    start_date := MAKE_DATE(p_year, 1, 1);
    end_date := start_date + INTERVAL '1 year';
    
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
        partition_name,
        p_table_name,
        start_date,
        end_date
    );
    
    RETURN partition_name;
END;
$$;
