# Market Data Warehouse

A robust historical data foundation for multi-market prediction systems, supporting both public markets (equities/ETFs) and event prediction markets (Polymarket-style contracts).

## Overview

This repository provides:

- **Raw Data Lake**: Partitioned storage for ingested data (Parquet/JSON)
- **SQL Warehouse**: Immutable raw tables + curated, point-in-time clean tables
- **Time-Correctness**: Every record carries `event_time` and `available_time` to prevent look-ahead bias
- **Backtest-Ready Snapshots**: Generate consistent as-of views for any market/contract at any time `t`
- **Data Quality**: Automated tests for time monotonicity, referential integrity, and anti-look-ahead
- **Strategy Engine**: Signal generation, position sizing, risk management, and backtesting
- **Execution Layer**: Live and paper trading via Alpaca, signal-to-order execution, capital guards, and position monitoring
- **Market Making**: Quoting, spread computation, inventory management, and hedging

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Data Sources  │────▶│  Raw Lake    │────▶│  Raw Tables     │
│  (FRED/GDELT/   │     │  (Parquet)   │     │  (Append-only)  │
│  Polymarket/    │     │              │     │                 │
│  Yahoo Finance/ │     │              │     │                 │
│  SEC EDGAR/     │     │              │     │                 │
│  Reddit/etc.)   │     │              │     │                 │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Training Data  │◀────│  Snapshots   │◀────│ Curated Tables  │
│                 │     │  (Point-in-  │     │  (Deduped,      │
│                 │     │   time)      │     │   Typed)        │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+ (3.12 also supported)
- PostgreSQL 16+ (or Docker)
- Docker & Docker Compose (recommended for local database)
- Make (optional, for convenience commands)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd finance-quant

# Install dependencies
pip install -e .

# Or install with development tools (recommended)
pip install -e ".[dev]"
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env
```

Edit `.env` and configure the following:

| Variable | Required | Description |
|----------|----------|-------------|
| `DB_HOST` | Yes | Database host (default: `localhost`) |
| `DB_PORT` | Yes | Database port (default: `5432`) |
| `DB_NAME` | Yes | Database name (default: `market_data`) |
| `DB_USER` | Yes | Database user (default: `postgres`) |
| `DB_PASSWORD` | Yes | Database password |
| `FRED_API_KEY` | Yes | API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `PRICES_SOURCE` | No | Price data provider: `yahoo` (default), `alphavantage`, or `polygon` |
| `PRICES_API_KEY` | No | Required if using `alphavantage` or `polygon` |
| `POLYMARKET_RATE_LIMIT_PER_SEC` | No | Rate limit for Polymarket API (default: `5.0`) |
| `ALPACA_API_KEY` | No | API key from [Alpaca](https://app.alpaca.markets/) (required for live/paper trading) |
| `ALPACA_SECRET_KEY` | No | Alpaca secret key (required for live/paper trading) |
| `ALPACA_BASE_URL` | No | Alpaca API URL (default: `https://paper-api.alpaca.markets`) |
| `NOTIFY_SLACK_WEBHOOK_URL` | No | Slack webhook for trade/alert notifications |
| `NOTIFY_SMTP_HOST` | No | SMTP host for email notifications |
| `NOTIFY_SMTP_PORT` | No | SMTP port (default: `587`) |
| `NOTIFY_SMTP_USER` | No | SMTP username |
| `NOTIFY_SMTP_PASSWORD` | No | SMTP password |
| `NOTIFY_EMAIL_FROM` | No | Sender email address |
| `NOTIFY_EMAIL_TO` | No | Recipient email address |
| `INFRA_MAX_ASYNC_WORKERS` | No | Max async workers (default: `10`) |
| `INFRA_BATCH_SIZE` | No | Batch size for data loading (default: `1000`) |
| `INFRA_SNAPSHOT_MAX_WORKERS` | No | Snapshot worker count (default: `4`) |

### Database Setup

**Option A: Using Make (recommended)**

```bash
# Start PostgreSQL container and initialize schema
make setup
```

This runs `make install`, `make db-up`, and `make db-init` in sequence.

**Option B: Using Docker Compose**

```bash
# Start PostgreSQL (auto-initializes schema from src/sql/ddl/)
docker compose up -d postgres

# Optional: start pgAdmin for database management
docker compose --profile admin up -d
# pgAdmin available at http://localhost:5050 (admin@localhost.com / admin)
```

**Option C: Manual**

```bash
# Start PostgreSQL container
docker run -d \
    --name mdw-postgres \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=market_data \
    -p 5432:5432 \
    postgres:16-alpine

# Initialize schema
python -m pipeline.cli init-db
```

### Verify Setup

```bash
# Run the test suite
make test

# Run linting (if dev dependencies installed)
make lint
```

### Run a Sample Pipeline

```bash
# Extract, load, transform, build snapshots, and run DQ checks
make full-pipeline
```

## Project Structure

```
finance-quant/
├── src/
│   ├── pipeline/
│   │   ├── cli.py                  # CLI entry point (Typer)
│   │   ├── settings.py             # Configuration management (Pydantic)
│   │   ├── db.py                   # Database utilities
│   │   ├── logging_config.py       # Logging setup
│   │   ├── extract/                # Data extractors
│   │   │   ├── fred.py             # FRED economic data
│   │   │   ├── gdelt.py            # GDELT world events
│   │   │   ├── polymarket.py       # Polymarket prediction markets
│   │   │   ├── prices_daily.py     # Daily OHLCV prices (Yahoo/etc.)
│   │   │   ├── sec_fundamentals.py # SEC EDGAR fundamentals
│   │   │   ├── sec_insider.py      # Insider trading data
│   │   │   ├── sec_13f.py          # Institutional holdings (13F)
│   │   │   ├── options_data.py     # Options market data
│   │   │   ├── earnings.py         # Earnings calendar
│   │   │   ├── reddit_sentiment.py # Social media sentiment
│   │   │   ├── short_interest.py   # Short interest data
│   │   │   ├── etf_flows.py        # ETF fund flows
│   │   │   └── factors_ff.py       # Fama-French factors
│   │   ├── load/                   # Raw data loaders
│   │   ├── transform/              # Curated transformations
│   │   ├── snapshot/               # Point-in-time snapshot builders
│   │   ├── dq/                     # Data quality tests
│   │   ├── features/               # Feature engineering
│   │   ├── backtesting/            # Backtesting engine
│   │   ├── strategy/               # Strategy signals & execution
│   │   ├── execution/              # Live/paper trading (Alpaca broker, signal executor)
│   │   ├── eval/                   # Performance evaluation & analysis
│   │   ├── market_making/          # Market making engine
│   │   ├── historical/             # Historical data management
│   │   └── infrastructure/         # Async pools, circuit breakers, metrics
│   └── sql/
│       └── ddl/                    # Database schema files (00-22)
├── tests/                          # Test suite
├── data/
│   └── raw/                        # Raw data lake (gitignored)
├── .env.example                    # Environment variable template
├── config.yaml                     # Pipeline configuration
├── docker-compose.yml              # Docker services
├── Makefile                        # Convenience commands
└── pyproject.toml                  # Package configuration
```

## Data Sources

| Source | Extractor | Description |
|--------|-----------|-------------|
| FRED | `fred.py` | 27 economic indicators (GDP, unemployment, CPI, credit spreads, FX, etc.) |
| GDELT | `gdelt.py` | Global database of world events |
| Polymarket | `polymarket.py` | Prediction market contracts, prices, trades, orderbooks |
| Yahoo Finance | `prices_daily.py` | Daily OHLCV prices for a configurable ticker universe |
| SEC EDGAR | `sec_fundamentals.py` | Company fundamentals from SEC filings |
| SEC Insider | `sec_insider.py` | Insider trading transactions |
| SEC 13F | `sec_13f.py` | Institutional holdings from 13F filings |
| Options | `options_data.py` | Options chain data |
| Earnings | `earnings.py` | Earnings calendar and results |
| Reddit | `reddit_sentiment.py` | Sentiment from r/wallstreetbets, r/stocks, r/investing, r/options |
| Short Interest | `short_interest.py` | Short interest data |
| ETF Flows | `etf_flows.py` | ETF fund flow data |
| Fama-French | `factors_ff.py` | Fama-French factor returns |

## Data Model

### Core Principle: Time Correctness

Every record carries two timestamps:

- **`event_time`**: When the thing happened in the real world
- **`available_time`**: Earliest time the system could have known it

**Rule**: Downstream training/backtests must only use records where `available_time <= t` for snapshot time `t`.

### Dimension Tables

| Table | Description |
|-------|-------------|
| `dim_source` | Data source registry |
| `dim_calendar_market` | Market calendars and trading days |
| `dim_entity` | Canonical entity registry (companies, countries, etc.) |
| `dim_symbol` | Financial instruments (stocks, ETFs, indices) |
| `dim_macro_series` | Economic indicator definitions |
| `dim_contract` | Prediction market contracts |

### Curated Fact Tables

| Table | Description |
|-------|-------------|
| `cur_prices_ohlcv_daily` | Daily OHLCV prices |
| `cur_macro_observations` | Economic data observations |
| `cur_world_events` | Structured world events (GDELT-style) |
| `cur_contract_prices` | Contract price history |
| `cur_contract_trades` | Individual trades |
| `cur_contract_orderbook_snapshots` | Orderbook snapshots |

### Snapshot Tables

| Table | Description |
|-------|-------------|
| `snap_contract_features` | Point-in-time contract features for training |

## CLI Usage

The pipeline CLI is available as `python -m pipeline.cli` or as the `mdw` command (after installation):

### Extract Data

```bash
# Extract FRED economic data
python -m pipeline.cli extract fred --start 2024-01-01 --end 2024-12-31

# Extract GDELT world events
python -m pipeline.cli extract gdelt --start 2024-11-01 --end 2024-11-30

# Extract market prices
python -m pipeline.cli extract prices --start 2024-01-01 --end 2024-12-31

# Extract Polymarket data
python -m pipeline.cli extract polymarket --start 2024-01-01 --end 2024-12-31
```

### Load and Transform

```bash
# Load raw files into database
python -m pipeline.cli load-raw fred

# Transform to curated tables
python -m pipeline.cli transform-curated
```

### Build Snapshots

```bash
# Build snapshots for all contracts
python -m pipeline.cli build-snapshots --start "2024-11-01T00:00:00" --end "2024-11-30T00:00:00" --freq 1h

# Build for specific contracts
python -m pipeline.cli build-snapshots --contract <uuid1> --contract <uuid2>
```

### Data Quality

```bash
# Run all DQ tests
python -m pipeline.cli dq
```

### Strategy & Execution

```bash
# Generate trading signals
python -m pipeline.cli generate-signals --strategy swing

# Execute signals (paper trading by default)
python -m pipeline.cli execute-signals --mode paper

# Check trading status (positions, P&L, orders)
python -m pipeline.cli trading-status

# Monitor live prices via WebSocket feed
python -m pipeline.cli monitor-prices --symbols SPY QQQ AAPL

# Monitor open positions
python -m pipeline.cli monitor-positions
```

### Evaluate Performance

```bash
# Run strategy evaluation
python -m pipeline.cli evaluate --strategy swing --start 2024-01-01 --end 2024-12-31
```

### Inventory

```bash
# Show data inventory
python -m pipeline.cli inventory
```

## Configuration

Edit `config.yaml` to customize data sources, ticker universes, and pipeline settings:

```yaml
# Database settings
database:
  host: localhost
  port: 5432
  name: market_data
  user: postgres
  password: postgres

# Source configurations
fred:
  series_codes:
    - GDP
    - UNRATE
    - CPIAUCSL

prices:
  universe:
    - SPY
    - QQQ
    - AAPL
```

Environment variables (from `.env`) override `config.yaml` values.

## Data Quality Tests

The pipeline includes comprehensive DQ tests:

| Test | Description |
|------|-------------|
| **Time Monotonicity** | `available_time >= event_time` for all records |
| **No Duplicate PKs** | Primary keys are unique in curated tables |
| **Referential Integrity** | Foreign keys reference valid records |
| **Coverage Sanity** | No negative prices/volumes; prices in valid range |
| **Snapshot Anti-Look-Ahead** | Snapshots don't include future data |

## Testing

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_dq.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make setup` | Full setup: install deps, start DB, init schema |
| `make install` | Install package dependencies |
| `make install-dev` | Install with dev dependencies (black, ruff, mypy, pytest) |
| `make db-up` | Start PostgreSQL via Docker |
| `make db-down` | Stop PostgreSQL container |
| `make db-init` | Initialize database schema |
| `make db-reset` | Reset database (stop, start, re-init) |
| `make extract-fred` | Extract and load FRED data |
| `make extract-gdelt` | Extract and load GDELT data |
| `make extract-prices` | Extract and load price data |
| `make extract-polymarket` | Extract and load Polymarket data |
| `make extract-all` | Extract and load all configured sources |
| `make transform` | Run curated transformations |
| `make snapshots` | Build training snapshots |
| `make full-pipeline` | Run complete pipeline (extract, transform, snapshot, DQ) |
| `make test` | Run all tests |
| `make test-dq` | Run data quality tests |
| `make test-snapshots` | Run snapshot tests |
| `make lint` | Run ruff and mypy checks |
| `make format` | Format code with black and ruff |
| `make inventory` | Show data inventory |
| `make clean` | Clean generated files and caches |
| `make docker-up` | Start all services via Docker Compose |
| `make docker-down` | Stop all Docker Compose services |

## Adding a New Source

1. **Create extractor** in `src/pipeline/extract/<source>.py`:

```python
class NewSourceExtractor:
    def extract_to_raw(self, output_dir: Path, ...) -> List[Path]:
        # Download and save to Parquet
        ...
```

2. **Create raw loader** in `src/pipeline/load/raw_loader.py`:

```python
def load_newsource_data(self, file_path: Path, run_id: UUID) -> int:
    # Load into raw_* table
    ...
```

3. **Create transformer** in `src/pipeline/transform/curated.py`:

```python
def transform_newsource_data(self) -> int:
    # Transform to cur_* table
    ...
```

4. **Add CLI command** in `src/pipeline/cli.py`:

```python
@app.command()
def extract_newsource(...):
    # Call extractor
    ...
```

## Engineering Notes

### Survivorship Bias

The schema supports delisted securities via `dim_symbol.is_delisted`. If you cannot ingest delisted equities initially, clearly label your dataset as survivorship-prone.

### Time Zones

All timestamps are stored in UTC. Original timezones are preserved in metadata where relevant.

### Resolution Rules

Store full rule text for prediction market contracts. Ambiguity is a major risk for later analysis.

### Schema Evolution

Database schema is managed via numbered DDL files in `src/sql/ddl/` (00 through 22). Files are applied in order. Add new migrations with the next available number.

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Make your changes
5. Format code: `make format`
6. Run linting: `make lint`
7. Run tests: `make test`
8. Submit a pull request

## Support

For issues and questions, please open a GitHub issue.
