"""Unit tests for data validation (no database required)."""

import numpy as np
import pandas as pd
import pytest

from pipeline.infrastructure.validation import (
    BatchValidator,
    ContractPriceValidator,
    FredObservationValidator,
    PriceValidator,
    ValidationResult,
    validate_fred_observations,
    validate_prices_ohlcv,
)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_is_valid_when_no_errors(self):
        result = ValidationResult()
        assert result.is_valid is True

    def test_is_invalid_when_has_errors(self):
        result = ValidationResult()
        result.add_error("something broke")
        assert result.is_valid is False

    def test_merge(self):
        r1 = ValidationResult()
        r1.add_error("err1")
        r1.add_warning("warn1")

        r2 = ValidationResult()
        r2.add_error("err2")
        r2.add_stat("count", 10)

        r1.merge(r2)
        assert len(r1.errors) == 2
        assert len(r1.warnings) == 1
        assert r1.stats["count"] == 10


class TestPriceValidator:
    """Tests for OHLCV price validation."""

    def test_valid_price(self):
        price = PriceValidator(
            ticker="SPY",
            date="2024-01-01",
            open=470.0,
            high=475.0,
            low=469.0,
            close=473.0,
            volume=1000000,
        )
        assert price.ticker == "SPY"

    def test_negative_price_rejected(self):
        with pytest.raises(Exception):
            PriceValidator(
                ticker="SPY",
                date="2024-01-01",
                open=-1.0,
                high=475.0,
                low=469.0,
                close=473.0,
                volume=1000000,
            )

    def test_high_less_than_open_rejected(self):
        with pytest.raises(Exception):
            PriceValidator(
                ticker="SPY",
                date="2024-01-01",
                open=475.0,
                high=470.0,  # less than open
                low=469.0,
                close=473.0,
                volume=1000000,
            )

    def test_low_greater_than_high_rejected(self):
        with pytest.raises(Exception):
            PriceValidator(
                ticker="SPY",
                date="2024-01-01",
                open=470.0,
                high=475.0,
                low=480.0,  # greater than high
                close=473.0,
                volume=1000000,
            )


class TestFredObservationValidator:
    """Tests for FRED observation validation."""

    def test_valid_observation(self):
        obs = FredObservationValidator(
            series_code="GDP",
            date="2024-01-01",
            value=28000.5,
        )
        assert obs.series_code == "GDP"

    def test_nan_value_becomes_none(self):
        obs = FredObservationValidator(
            series_code="GDP",
            date="2024-01-01",
            value=float("nan"),
        )
        assert obs.value is None

    def test_empty_series_code_rejected(self):
        with pytest.raises(Exception):
            FredObservationValidator(
                series_code="",
                date="2024-01-01",
                value=100.0,
            )


class TestContractPriceValidator:
    """Tests for contract price validation."""

    def test_valid_contract_price(self):
        price = ContractPriceValidator(
            contract_id="abc-123",
            ts="2024-01-01T00:00:00",
            outcome="YES",
            price_normalized=0.55,
            price_raw=55.0,
        )
        assert price.price_normalized == 0.55

    def test_normalized_price_out_of_range(self):
        with pytest.raises(Exception):
            ContractPriceValidator(
                contract_id="abc-123",
                ts="2024-01-01T00:00:00",
                outcome="YES",
                price_normalized=1.5,
                price_raw=150.0,
            )


class TestBatchValidator:
    """Tests for batch validation."""

    def test_all_valid(self):
        validator = BatchValidator(FredObservationValidator)
        records = [
            {"series_code": "GDP", "date": "2024-01-01", "value": 100.0},
            {"series_code": "GDP", "date": "2024-02-01", "value": 101.0},
        ]

        valid, result = validator.validate_batch(records)

        assert len(valid) == 2
        assert result.is_valid

    def test_partial_invalid(self):
        validator = BatchValidator(FredObservationValidator)
        records = [
            {"series_code": "GDP", "date": "2024-01-01", "value": 100.0},
            {"series_code": "", "date": "2024-02-01", "value": 101.0},  # invalid
        ]

        valid, result = validator.validate_batch(records)

        assert len(valid) == 1
        assert not result.is_valid

    def test_max_errors_stops_early(self):
        validator = BatchValidator(FredObservationValidator, max_errors=2)
        records = [
            {"series_code": "", "date": "2024-01-01", "value": 100.0},
            {"series_code": "", "date": "2024-02-01", "value": 101.0},
            {"series_code": "", "date": "2024-03-01", "value": 102.0},
        ]

        valid, result = validator.validate_batch(records)

        assert len(valid) == 0
        # Should stop at max_errors (2) + 1 for the "max errors reached" message
        assert len(result.errors) == 3


class TestValidateFredObservations:
    """Tests for DataFrame-level FRED validation."""

    def test_valid_dataframe(self):
        df = pd.DataFrame({
            "series_code": ["GDP", "GDP"],
            "date": ["2024-01-01", "2024-02-01"],
            "value": [100.0, 101.0],
        })

        result = validate_fred_observations(df)
        assert result.is_valid

    def test_missing_columns(self):
        df = pd.DataFrame({"series_code": ["GDP"]})

        result = validate_fred_observations(df)
        assert not result.is_valid
        assert "Missing columns" in result.errors[0]

    def test_null_series_code(self):
        df = pd.DataFrame({
            "series_code": [None],
            "date": ["2024-01-01"],
            "value": [100.0],
        })

        result = validate_fred_observations(df)
        assert not result.is_valid


class TestValidatePricesOhlcv:
    """Tests for DataFrame-level OHLCV validation."""

    def test_valid_prices(self):
        df = pd.DataFrame({
            "open": [470.0],
            "high": [475.0],
            "low": [469.0],
            "close": [473.0],
            "volume": [1000000],
        })

        result = validate_prices_ohlcv(df)
        assert result.is_valid

    def test_low_greater_than_high(self):
        df = pd.DataFrame({
            "open": [470.0],
            "high": [475.0],
            "low": [480.0],  # violation
            "close": [473.0],
            "volume": [1000000],
        })

        result = validate_prices_ohlcv(df)
        assert not result.is_valid

    def test_negative_volume(self):
        df = pd.DataFrame({
            "open": [470.0],
            "high": [475.0],
            "low": [469.0],
            "close": [473.0],
            "volume": [-100],
        })

        result = validate_prices_ohlcv(df)
        assert not result.is_valid
