-- Raw tables for SEC EDGAR data sources

-- Raw SEC fundamentals (XBRL company facts)
CREATE TABLE IF NOT EXISTS raw_sec_fundamentals (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20) NOT NULL,
    cik INTEGER NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_label VARCHAR(200),
    metric_value NUMERIC,
    units VARCHAR(50),
    fiscal_period_end DATE NOT NULL,
    filing_date DATE NOT NULL,
    form_type VARCHAR(10),
    accession_number VARCHAR(30),
    fiscal_year INTEGER,
    fiscal_period VARCHAR(10),
    raw_data JSONB,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(ticker, metric_name, fiscal_period_end, form_type, accession_number)
);

-- Restatement tracking: detect amended filings and order by filing sequence.
ALTER TABLE raw_sec_fundamentals ADD COLUMN IF NOT EXISTS is_amendment BOOLEAN DEFAULT FALSE;
ALTER TABLE raw_sec_fundamentals ADD COLUMN IF NOT EXISTS original_form_type VARCHAR(10);
ALTER TABLE raw_sec_fundamentals ADD COLUMN IF NOT EXISTS filing_sequence INTEGER DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_raw_sec_fund_ticker ON raw_sec_fundamentals(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_sec_fund_filing ON raw_sec_fundamentals(filing_date);
CREATE INDEX IF NOT EXISTS idx_raw_sec_fund_period ON raw_sec_fundamentals(fiscal_period_end);

-- Raw SEC insider trades (Form 4)
CREATE TABLE IF NOT EXISTS raw_sec_insider_trades (
    id BIGSERIAL PRIMARY KEY,
    ticker VARCHAR(20),
    cik INTEGER NOT NULL,
    insider_cik INTEGER,
    insider_name VARCHAR(200),
    insider_title VARCHAR(100),
    transaction_date DATE,
    transaction_type VARCHAR(10),
    shares NUMERIC,
    price_per_share NUMERIC,
    shares_after NUMERIC,
    ownership_type VARCHAR(1),
    form_type VARCHAR(10) DEFAULT '4',
    accession_number VARCHAR(30),
    filing_date DATE NOT NULL,
    raw_data JSONB,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(accession_number, insider_cik, transaction_date, transaction_type, shares)
);

CREATE INDEX IF NOT EXISTS idx_raw_sec_insider_ticker ON raw_sec_insider_trades(ticker);
CREATE INDEX IF NOT EXISTS idx_raw_sec_insider_filing ON raw_sec_insider_trades(filing_date);
CREATE INDEX IF NOT EXISTS idx_raw_sec_insider_txn ON raw_sec_insider_trades(transaction_date);

-- Raw SEC 13F institutional holdings
CREATE TABLE IF NOT EXISTS raw_sec_13f_holdings (
    id BIGSERIAL PRIMARY KEY,
    filer_cik INTEGER NOT NULL,
    filer_name VARCHAR(300),
    report_date DATE NOT NULL,
    filing_date DATE NOT NULL,
    cusip VARCHAR(9),
    issuer_name VARCHAR(300),
    class_title VARCHAR(100),
    market_value BIGINT,
    shares_held BIGINT,
    shares_type VARCHAR(10),
    put_call VARCHAR(10),
    investment_discretion VARCHAR(10),
    voting_authority_sole BIGINT,
    voting_authority_shared BIGINT,
    voting_authority_none BIGINT,
    accession_number VARCHAR(30),
    raw_data JSONB,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id VARCHAR(36),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(accession_number, cusip, filer_cik, report_date)
);

CREATE INDEX IF NOT EXISTS idx_raw_sec_13f_filer ON raw_sec_13f_holdings(filer_cik);
CREATE INDEX IF NOT EXISTS idx_raw_sec_13f_cusip ON raw_sec_13f_holdings(cusip);
CREATE INDEX IF NOT EXISTS idx_raw_sec_13f_report ON raw_sec_13f_holdings(report_date);
CREATE INDEX IF NOT EXISTS idx_raw_sec_13f_filing ON raw_sec_13f_holdings(filing_date);
