.PHONY: help setup install install-dev db-init db-reset test test-dq lint format clean extract-all transform snapshots inventory full-pipeline daily-predictions build-site extract-fred extract-gdelt extract-prices extract-polymarket extract-sec-fundamentals extract-sec-insider extract-sec-13f extract-options extract-earnings extract-reddit extract-short-interest extract-etf-flows extract-cftc extract-factors

# Default target
help:
	@echo "Market Data Warehouse - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Full setup: install deps + init DuckDB schema"
	@echo "  make install        - Install package dependencies"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo ""
	@echo "Database:"
	@echo "  make db-init        - Initialize database schema"
	@echo "  make db-reset       - Delete and reinitialize DuckDB"
	@echo ""
	@echo "Database (PostgreSQL - optional):"
	@echo "  make pg-up          - Start PostgreSQL via Docker"
	@echo "  make pg-down        - Stop PostgreSQL"
	@echo "  make pg-init        - Initialize PostgreSQL schema"
	@echo ""
	@echo "Pipeline:"
	@echo "  make extract-all    - Extract all configured sources"
	@echo "  make transform      - Run curated transformations"
	@echo "  make snapshots      - Build training snapshots"
	@echo "  make full-pipeline  - Run complete pipeline"
	@echo ""
	@echo "Quality:"
	@echo "  make test           - Run all tests"
	@echo "  make test-dq        - Run data quality tests"
	@echo "  make lint           - Run linting"
	@echo "  make format         - Format code"
	@echo ""
	@echo "Utilities:"
	@echo "  make inventory      - Show data inventory"
	@echo "  make clean          - Clean generated files"

# Setup (no Docker needed!)
setup: install db-init
	@echo "✓ Setup complete (using DuckDB — no server required)"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Database — DuckDB (default, zero-config)
db-init:
	@echo "Initializing DuckDB schema..."
	python -m pipeline.cli init-db
	@echo "✓ Schema initialized"

db-reset:
	@echo "Resetting DuckDB..."
	rm -f data/market_data.duckdb data/market_data.duckdb.wal
	$(MAKE) db-init

# Database — PostgreSQL (optional, for production)
pg-up:
	@echo "Starting PostgreSQL..."
	docker compose up -d postgres
	@sleep 3
	@echo "✓ PostgreSQL started"

pg-down:
	docker compose down
	@echo "✓ PostgreSQL stopped"

pg-init:
	@echo "Initializing PostgreSQL schema..."
	DB_BACKEND=postgresql python -m pipeline.cli init-db
	@echo "✓ PostgreSQL schema initialized"

pg-reset: pg-down
	@sleep 2
	$(MAKE) pg-up
	@sleep 3
	$(MAKE) pg-init

# Pipeline commands
extract-fred:
	python -m pipeline.cli extract fred --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw fred

extract-gdelt:
	python -m pipeline.cli extract gdelt --start 2024-11-01 --end 2024-11-30
	python -m pipeline.cli load-raw gdelt

extract-prices:
	python -m pipeline.cli extract prices --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw prices

extract-polymarket:
	python -m pipeline.cli extract polymarket --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw polymarket

extract-sec-fundamentals:
	python -m pipeline.cli extract sec-fundamentals --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw sec_fundamentals

extract-sec-insider:
	python -m pipeline.cli extract sec-insider --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw sec_insider

extract-sec-13f:
	python -m pipeline.cli extract sec-13f --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw sec_13f

extract-options:
	python -m pipeline.cli extract options --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw options

extract-earnings:
	python -m pipeline.cli extract earnings --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw earnings

extract-reddit:
	python -m pipeline.cli extract reddit-sentiment --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw reddit_sentiment

extract-short-interest:
	python -m pipeline.cli extract short-interest --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw short_interest

extract-etf-flows:
	python -m pipeline.cli extract etf-flows --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw etf_flows

extract-cftc:
	python -m pipeline.cli extract cftc-cot --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw cftc_cot

extract-factors:
	python -m pipeline.cli extract factors --start 2024-01-01 --end 2024-12-31
	python -m pipeline.cli load-raw factors

# Prices are required; other sources are best-effort (may lack API keys in CI)
extract-all: extract-prices
	-$(MAKE) extract-fred
	-$(MAKE) extract-gdelt
	-$(MAKE) extract-polymarket
	-$(MAKE) extract-sec-fundamentals
	-$(MAKE) extract-sec-insider
	-$(MAKE) extract-sec-13f
	-$(MAKE) extract-options
	-$(MAKE) extract-earnings
	-$(MAKE) extract-reddit
	-$(MAKE) extract-short-interest
	-$(MAKE) extract-etf-flows
	-$(MAKE) extract-cftc
	-$(MAKE) extract-factors
	@echo "✓ Extraction complete (prices required; other sources best-effort)"

transform: db-init
	python -m pipeline.cli transform-curated

snapshots:
	python -m pipeline.cli build-snapshots --start "2024-11-01T00:00:00" --end "2024-11-30T00:00:00" --freq 1d

dq:
	python -m pipeline.cli dq

inventory:
	python -m pipeline.cli inventory

full-pipeline: extract-all transform snapshots dq inventory
	@echo "✓ Full pipeline complete"

# Daily predictions
daily-predictions:
	python -m pipeline.cli daily-predictions
	@echo "✓ Daily predictions complete — view site/index.html"

build-site:
	python -m pipeline.cli daily-predictions --date $(shell date +%Y-%m-%d)
	@echo "✓ Site built"

# Testing
test:
	pytest tests/ -v

test-dq:
	python -m pipeline.cli dq

test-snapshots:
	pytest tests/test_snapshots.py -v

# Code quality
lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

# Cleanup
clean:
	rm -rf data/raw/*
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned"

# Docker Compose alternative (PostgreSQL)
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
