"""Unit tests for technical indicators (no database required)."""

import numpy as np
import pandas as pd
import pytest

from pipeline.features.technical_indicators import (
    ContractFeatureEngineer,
    TechnicalIndicators,
)


@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    np.random.seed(42)
    n = 100
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.Series(prices, name="close")


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": close + np.abs(np.random.randn(n) * 0.5),
        "low": close - np.abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(100000, 1000000, n).astype(float),
    })


class TestSMA:
    def test_sma_length(self, sample_prices):
        result = TechnicalIndicators.sma(sample_prices, 10)
        assert len(result) == len(sample_prices)

    def test_sma_values(self):
        prices = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = TechnicalIndicators.sma(prices, 3)
        # With min_periods=1: [1, 1.5, 2, 3, 4]
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[4] == pytest.approx(4.0)


class TestEMA:
    def test_ema_length(self, sample_prices):
        result = TechnicalIndicators.ema(sample_prices, 12)
        assert len(result) == len(sample_prices)


class TestRSI:
    def test_rsi_range(self, sample_prices):
        result = TechnicalIndicators.rsi(sample_prices, 14)
        assert result.min() >= 0
        assert result.max() <= 100

    def test_rsi_length(self, sample_prices):
        result = TechnicalIndicators.rsi(sample_prices, 14)
        assert len(result) == len(sample_prices)


class TestMACD:
    def test_macd_returns_three_series(self, sample_prices):
        macd, signal, hist = TechnicalIndicators.macd(sample_prices)
        assert len(macd) == len(sample_prices)
        assert len(signal) == len(sample_prices)
        assert len(hist) == len(sample_prices)

    def test_histogram_is_macd_minus_signal(self, sample_prices):
        macd, signal, hist = TechnicalIndicators.macd(sample_prices)
        diff = macd - signal
        pd.testing.assert_series_equal(hist, diff)


class TestBollingerBands:
    def test_bands_relationship(self, sample_prices):
        upper, middle, lower = TechnicalIndicators.bollinger_bands(sample_prices, 20)
        # Drop NaN values from rolling window warmup before comparing
        mask = upper.notna() & middle.notna() & lower.notna()
        assert (upper[mask] >= middle[mask]).all()
        assert (middle[mask] >= lower[mask]).all()


class TestATR:
    def test_atr_non_negative(self, sample_ohlcv):
        result = TechnicalIndicators.atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            14,
        )
        assert (result >= 0).all()


class TestStochastic:
    def test_stochastic_range(self, sample_ohlcv):
        k, d = TechnicalIndicators.stochastic(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )
        assert k.min() >= 0
        assert k.max() <= 100


class TestCalculateAll:
    def test_adds_all_indicator_columns(self, sample_ohlcv):
        result = TechnicalIndicators.calculate_all(sample_ohlcv)

        expected_cols = [
            "sma_10", "sma_20", "sma_50",
            "ema_12", "ema_26",
            "rsi_14", "momentum_10", "roc_10",
            "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_position",
            "atr_14", "stoch_k", "stoch_d", "williams_r",
            "obv", "volume_sma_20",
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_has_same_length(self, sample_ohlcv):
        result = TechnicalIndicators.calculate_all(sample_ohlcv)
        assert len(result) == len(sample_ohlcv)


class TestContractFeatureEngineer:
    def test_price_features(self):
        prices = pd.Series([0.5, 0.52, 0.48, 0.55, 0.53])
        features = ContractFeatureEngineer.calculate_price_features(prices)

        assert "price" in features.columns
        assert "price_change_1h" in features.columns
        assert "price_volatility_24h" in features.columns

    def test_liquidity_features_empty(self):
        empty = pd.DataFrame()
        result = ContractFeatureEngineer.calculate_liquidity_features(empty)
        assert result.empty
