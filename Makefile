.PHONY: help setup install install-dev db-up db-down db-init test test-dq lint format clean extract-all transform snapshots inventory full-pipeline

# Default target
help:
	@echo "Market Data Warehouse - Available Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Full setup: install deps, start DB, init schema"
	@echo "  make install        - Install package dependencies"
	@echo "  make install-dev    - Install with dev dependencies"
	@echo ""
	@echo "Database:"
	@echo "  make db-up          - Start PostgreSQL via Docker"
	@echo "  make db-down        - Stop PostgreSQL"
	@echo "  make db-init        - Initialize database schema"
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

# Setup
setup: install db-up db-init
	@echo "✓ Setup complete"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Database (Docker)
db-up:
	@echo "Starting PostgreSQL..."
	docker run -d \
		--name mdw-postgres \
		-e POSTGRES_USER=postgres \
		-e POSTGRES_PASSWORD=postgres \
		-e POSTGRES_DB=market_data \
		-p 5432:5432 \
		postgres:16-alpine || true
	@sleep 3
	@echo "✓ PostgreSQL started"

db-down:
	docker stop mdw-postgres || true
	docker rm mdw-postgres || true
	@echo "✓ PostgreSQL stopped"

db-init:
	@echo "Initializing database schema..."
	python -m pipeline.cli init-db
	@echo "✓ Schema initialized"

db-reset: db-down
	@sleep 2
	$(MAKE) db-up
	@sleep 3
	$(MAKE) db-init

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

extract-all: extract-fred extract-gdelt extract-prices extract-polymarket
	@echo "✓ All sources extracted"

transform:
	python -m pipeline.cli transform-curated

snapshots:
	python -m pipeline.cli build-snapshots --start "2024-11-01T00:00:00" --end "2024-11-30T00:00:00" --freq 1d

dq:
	python -m pipeline.cli dq

inventory:
	python -m pipeline.cli inventory

full-pipeline: extract-all transform snapshots dq inventory
	@echo "✓ Full pipeline complete"

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

# Docker Compose alternative
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
