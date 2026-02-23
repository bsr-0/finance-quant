"""Data validation framework for ingestion pipeline."""

import logging
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ValidationResult:
    """Result of data validation."""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict = {}

    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return len(self.errors) == 0

    def add_error(self, message: str):
        """Add error message."""
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add warning message."""
        self.warnings.append(message)

    def add_stat(self, key: str, value: Any):
        """Add statistic."""
        self.stats[key] = value

    def merge(self, other: "ValidationResult"):
        """Merge another validation result."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.stats.update(other.stats)


class FredObservationValidator(BaseModel):
    """Validator for FRED observations."""

    series_code: str = Field(..., min_length=1)
    date: str
    value: float | None = None

    @field_validator("value")
    @classmethod
    def validate_value(cls, v):
        if v is not None and (v != v):  # NaN check
            return None
        return v


class PriceValidator(BaseModel):
    """Validator for OHLCV prices."""

    ticker: str = Field(..., min_length=1, max_length=20)
    date: str
    open: float = Field(..., ge=0)
    high: float = Field(..., ge=0)
    low: float = Field(..., ge=0)
    close: float = Field(..., ge=0)
    volume: float = Field(..., ge=0)

    @field_validator("high")
    @classmethod
    def high_gte_open(cls, v, info):
        values = info.data
        if "open" in values and v < values["open"]:
            raise ValueError("high must be >= open")
        return v

    @field_validator("low")
    @classmethod
    def low_lte_high(cls, v, info):
        values = info.data
        if "high" in values and v > values["high"]:
            raise ValueError("low must be <= high")
        return v

    @field_validator("close")
    @classmethod
    def close_in_range(cls, v, info):
        values = info.data
        if "low" in values and "high" in values and (v < values["low"] or v > values["high"]):
            raise ValueError("close must be between low and high")
        return v


class ContractPriceValidator(BaseModel):
    """Validator for contract prices."""

    contract_id: str
    ts: str
    outcome: str
    price_normalized: float = Field(..., ge=0, le=1)
    price_raw: float = Field(..., ge=0)


class DataValidator:
    """Generic data validator with multiple validation rules."""

    def __init__(self):
        self.rules: list[Callable[[Any], ValidationResult]] = []

    def add_rule(self, rule: Callable[[Any], ValidationResult]):
        """Add validation rule."""
        self.rules.append(rule)

    def validate(self, data: Any) -> ValidationResult:
        """Run all validation rules."""
        result = ValidationResult()

        for rule in self.rules:
            try:
                rule_result = rule(data)
                result.merge(rule_result)
            except Exception as e:
                result.add_error(f"Validation rule failed: {e}")

        return result


class BatchValidator:
    """Validate batches of records with detailed reporting."""

    def __init__(self, validator_class: type[BaseModel], max_errors: int = 100):
        self.validator_class = validator_class
        self.max_errors = max_errors

    def validate_batch(
        self, records: list[dict], fail_fast: bool = False
    ) -> tuple[list[dict], ValidationResult]:
        """Validate a batch of records.

        Returns:
            Tuple of (valid_records, validation_result)
        """
        result = ValidationResult()
        valid_records = []

        for i, record in enumerate(records):
            try:
                validated = self.validator_class(**record)
                valid_records.append(validated.model_dump())
            except ValidationError as e:
                error_msg = f"Record {i}: {e.errors()[0]['msg']}"
                result.add_error(error_msg)

                if len(result.errors) >= self.max_errors:
                    result.add_error(f"Max errors ({self.max_errors}) reached, stopping validation")
                    break

                if fail_fast:
                    break

        result.add_stat("total_records", len(records))
        result.add_stat("valid_records", len(valid_records))
        result.add_stat("invalid_records", len(records) - len(valid_records))

        return valid_records, result


def validate_fred_observations(df) -> ValidationResult:
    """Validate FRED observations DataFrame."""
    result = ValidationResult()

    # Check required columns
    required_cols = ["series_code", "date", "value"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        result.add_error(f"Missing columns: {missing_cols}")
        return result

    # Check for null values in key columns
    null_counts = df[["series_code", "date"]].isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            result.add_error(f"Column '{col}' has {count} null values")

    # Check value ranges
    if "value" in df.columns:
        valid_values = df["value"].notna()
        result.add_stat("valid_value_count", valid_values.sum())
        result.add_stat("null_value_count", (~valid_values).sum())

    # Check date format
    try:
        pd_dates = pd.to_datetime(df["date"], errors="coerce")
        invalid_dates = pd_dates.isna().sum()
        if invalid_dates > 0:
            result.add_error(f"{invalid_dates} records have invalid dates")
    except Exception as e:
        result.add_error(f"Date validation failed: {e}")

    return result


def validate_prices_ohlcv(df) -> ValidationResult:
    """Validate OHLCV prices DataFrame."""
    result = ValidationResult()

    # OHLC logic validation
    ohlc_violations = (
        (df["low"] > df["high"])
        | (df["open"] > df["high"])
        | (df["open"] < df["low"])
        | (df["close"] > df["high"])
        | (df["close"] < df["low"])
    )

    violation_count = ohlc_violations.sum()
    if violation_count > 0:
        result.add_error(f"{violation_count} records violate OHLC logic")

    # Negative price check
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        negative = (df[col] < 0).sum()
        if negative > 0:
            result.add_error(f"{negative} records have negative {col}")

    # Volume check
    negative_volume = (df["volume"] < 0).sum()
    if negative_volume > 0:
        result.add_error(f"{negative_volume} records have negative volume")

    # Statistics
    result.add_stat("total_records", len(df))
    result.add_stat("avg_volume", df["volume"].mean())
    result.add_stat("price_range", df["high"].max() - df["low"].min())

    return result
