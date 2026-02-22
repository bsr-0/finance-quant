# Market Data Warehouse

A robust historical data foundation for multi-market prediction systems, supporting both public markets (equities/ETFs) and event prediction markets (Polymarket-style contracts).

## Overview

This repository provides:

- **Raw Data Lake**: Partitioned storage for ingested data (Parquet/JSON)
- **SQL Warehouse**: Immutable raw tables + curated, point-in-time clean tables
- **Time-Correctness**: Every record carries `event_time` and `available_time` to prevent look-ahead bias
- **Backtest-Ready Snapshots**: Generate consistent as-of views for any market/contract at any time `t`
- **Data Quality**: Automated tests for time monotonicity, referential integrity, and anti-look-ahead

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Data Sources  │────▶│  Raw Lake    │────▶│  Raw Tables     │
│  (FRED/GDELT/   │     │  (Parquet)   │     │  (Append-only)  │
│  Polymarket/    │     │              │     │                 │
│  Yahoo Finance) │     │              │     │                 │
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

- Python 3.11+
- PostgreSQL 14+ (or Docker)
- Make (optional, for convenience commands)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd market-data-warehouse

# Install dependencies
make install
# Or: pip install -e .

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Database Setup

```bash
# Start PostgreSQL via Docker
make db-up

# Initialize schema
make db-init
```

Or use Docker Compose:

```bash
docker-compose up -d postgres
```

### Run Sample Pipeline

```bash
# Full setup and sample data load
make setup

# Run complete pipeline
make full-pipeline
```

## Project Structure

```
market-data-warehouse/
├── src/
│   ├── pipeline/
│   │   ├── cli.py              # Command-line interface
│   │   ├── settings.py         # Configuration management
│   │   ├── db.py               # Database utilities
│   │   ├── extract/            # Data extractors
│   │   │   ├── fred.py         # FRED economic data
│   │   │   ├── gdelt.py        # GDELT world events
│   │   │   ├── polymarket.py   # Polymarket prediction markets
│   │   │   └── prices_daily.py # Daily OHLCV prices
│   │   ├── load/               # Raw data loaders
│   │   ├── transform/          # Curated transformations
│   │   ├── snapshot/           # Snapshot builders
│   │   └── dq/                 # Data quality tests
│   └── sql/
│       └── ddl/                # Database schema files
├── tests/                      # Test suite
├── data/
│   └── raw/                    # Raw data lake (gitignored)
├── config.yaml                 # Pipeline configuration
├── docker-compose.yml          # Docker services
├── Makefile                    # Convenience commands
└── pyproject.toml              # Package configuration
```

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

# Or use make
make test-dq
```

### Inventory

```bash
# Show data inventory
python -m pipeline.cli inventory
```

## Configuration

Edit `config.yaml` to customize:

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

Or use environment variables:

```bash
export DB_HOST=localhost
export DB_PASSWORD=secret
export FRED_API_KEY=your_key_here
```

## Data Quality Tests

The pipeline includes comprehensive DQ tests:

| Test | Description |
|------|-------------|
| **Time Monotonicity** | `available_time >= event_time` for all records |
| **No Duplicate PKs** | Primary keys are unique in curated tables |
| **Referential Integrity** | Foreign keys reference valid records |
| **Coverage Sanity** | No negative prices/volumes; prices in valid range |
| **Snapshot Anti-Look-Ahead** | Snapshots don't include future data |

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

## Snapshot Builder

The snapshot builder creates point-in-time features for training:

```python
from pipeline.snapshot.contract_snapshots import ContractSnapshotBuilder

builder = ContractSnapshotBuilder()
snapshot = builder.build_contract_snapshot(
    contract_id=uuid,
    asof_ts=datetime(2024, 11, 1, 12, 0, 0)
)

# Returns:
# {
#     "contract_id": uuid,
#     "asof_ts": datetime,
#     "implied_p_yes": 0.65,
#     "spread": 0.02,
#     "volume_24h": 15000.0,
#     "macro_panel": {"UNRATE": 4.1, "GDP": ...},
#     "news_counts": {"1h": 5, "24h": 45},
#     ...
# }
```

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
| `make setup` | Full setup: install deps, start DB, init schema |
| `make db-up` | Start PostgreSQL via Docker |
| `make db-init` | Initialize database schema |
| `make extract-all` | Extract all configured sources |
| `make transform` | Run curated transformations |
| `make snapshots` | Build training snapshots |
| `make full-pipeline` | Run complete pipeline |
| `make test` | Run all tests |
| `make dq` | Run data quality tests |
| `make inventory` | Show data inventory |
| `make clean` | Clean generated files |

## Engineering Notes

### Survivorship Bias

The schema supports delisted securities via `dim_symbol.is_delisted`. If you cannot ingest delisted equities initially, clearly label your dataset as survivorship-prone.

### Time Zones

All timestamps are stored in UTC. Original timezones are preserved in metadata where relevant.

### Resolution Rules

Store full rule text for prediction market contracts. Ambiguity is a major risk for later analysis.

### Schema Evolution

Add `schema_version` fields to raw tables or maintain migration scripts for schema changes.

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Submit a pull request

## Support

For issues and questions, please open a GitHub issue.
