"""Tests for multi-source price extraction, corporate actions adjustment,
and walk-forward strategy validation."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Corporate actions adjustment tests
# ---------------------------------------------------------------------------


class TestCorporateActionsAdjustment:
    """Tests for adjust_for_corporate_actions and _parse_split_ratio."""

    def test_parse_split_ratio_string(self):
        from pipeline.extract.prices_daily import _parse_split_ratio

        assert _parse_split_ratio("4:1") == 4.0
        assert _parse_split_ratio("1:1") == 1.0
        assert _parse_split_ratio("2:1") == 2.0
        assert _parse_split_ratio("3:2") == 1.5

    def test_parse_split_ratio_numeric(self):
        from pipeline.extract.prices_daily import _parse_split_ratio

        assert _parse_split_ratio(4.0) == 4.0
        assert _parse_split_ratio(0) == 1.0

    def test_parse_split_ratio_invalid(self):
        from pipeline.extract.prices_daily import _parse_split_ratio

        assert _parse_split_ratio("bad") == 1.0
        assert _parse_split_ratio("0:0") == 1.0

    def test_adjust_empty_dataframe(self):
        from pipeline.extract.prices_daily import adjust_for_corporate_actions

        df = pd.DataFrame()
        result = adjust_for_corporate_actions(df)
        assert result.empty

    def test_adjust_no_actions(self):
        from pipeline.extract.prices_daily import adjust_for_corporate_actions

        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5).date,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [105.0, 106.0, 107.0, 108.0, 109.0],
                "low": [95.0, 96.0, 97.0, 98.0, 99.0],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "volume": [1000, 1000, 1000, 1000, 1000],
                "split_ratio": [None, None, None, None, None],
                "dividend": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        result = adjust_for_corporate_actions(df)
        # No adjustments needed — prices should remain the same
        np.testing.assert_array_almost_equal(result["close"].values, df["close"].values)

    def test_adjust_for_split(self):
        from pipeline.extract.prices_daily import adjust_for_corporate_actions

        # 2:1 split on day 3 — pre-split prices should be halved
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5).date,
                "open": [100.0, 100.0, 100.0, 50.0, 50.0],
                "high": [110.0, 110.0, 110.0, 55.0, 55.0],
                "low": [90.0, 90.0, 90.0, 45.0, 45.0],
                "close": [100.0, 100.0, 100.0, 50.0, 50.0],
                "volume": [1000, 1000, 1000, 2000, 2000],
                "split_ratio": [None, None, None, "2:1", None],
                "dividend": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        result = adjust_for_corporate_actions(df)

        # Pre-split closes (days 0-2) should be adjusted to ~50
        assert result["close"].iloc[0] == pytest.approx(50.0, abs=0.1)
        assert result["close"].iloc[1] == pytest.approx(50.0, abs=0.1)
        assert result["close"].iloc[2] == pytest.approx(50.0, abs=0.1)
        # Post-split prices stay the same
        assert result["close"].iloc[3] == pytest.approx(50.0, abs=0.1)
        assert result["close"].iloc[4] == pytest.approx(50.0, abs=0.1)

    def test_adjust_preserves_unadjusted_close(self):
        from pipeline.extract.prices_daily import adjust_for_corporate_actions

        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3).date,
                "open": [100.0, 100.0, 50.0],
                "high": [110.0, 110.0, 55.0],
                "low": [90.0, 90.0, 45.0],
                "close": [100.0, 100.0, 50.0],
                "volume": [1000, 1000, 2000],
                "split_ratio": [None, None, "2:1"],
                "dividend": [0.0, 0.0, 0.0],
            }
        )
        result = adjust_for_corporate_actions(df)
        assert "unadjusted_close" in result.columns
        np.testing.assert_array_almost_equal(
            result["unadjusted_close"].values, [100.0, 100.0, 50.0]
        )

    def test_adjust_for_dividend(self):
        from pipeline.extract.prices_daily import adjust_for_corporate_actions

        # $2 dividend on day 2 (ex-date), close was $100 before
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3).date,
                "open": [100.0, 100.0, 98.0],
                "high": [105.0, 105.0, 103.0],
                "low": [95.0, 95.0, 93.0],
                "close": [100.0, 100.0, 98.0],
                "volume": [1000, 1000, 1000],
                "split_ratio": [None, None, None],
                "dividend": [0.0, 0.0, 2.0],
            }
        )
        result = adjust_for_corporate_actions(df)
        # Day 0 close should be adjusted down by the dividend factor
        expected_factor = (100.0 - 2.0) / 100.0  # 0.98
        assert result["close"].iloc[0] == pytest.approx(100.0 * expected_factor, abs=0.1)


# ---------------------------------------------------------------------------
# Multi-source extractor tests
# ---------------------------------------------------------------------------


class TestPriceExtractorFallback:
    """Tests for PriceExtractor with fallback logic."""

    @patch("pipeline.extract.prices_daily.get_settings")
    def test_creates_yahoo_extractor_by_default(self, mock_settings):
        from pipeline.extract.prices_daily import PriceExtractor

        mock_cfg = MagicMock()
        mock_cfg.prices.source = "yahoo"
        mock_cfg.prices.fallback_source = None
        mock_cfg.prices.universe = ["SPY"]
        mock_cfg.prices.adjust_corporate_actions = True
        mock_cfg.prices.alpaca_api_key = None
        mock_cfg.prices.alpaca_secret_key = None
        mock_cfg.prices.polygon_api_key = None
        mock_settings.return_value = mock_cfg

        extractor = PriceExtractor()
        assert extractor.source == "yahoo"
        assert extractor._fallback_extractor is None

    @patch("pipeline.extract.prices_daily.get_settings")
    def test_unsupported_source_raises(self, mock_settings):
        from pipeline.extract.prices_daily import _create_extractor

        with pytest.raises(ValueError, match="Unsupported price source"):
            _create_extractor("nonexistent")

    def test_extractor_factory_known_sources(self):
        from pipeline.extract.prices_daily import _EXTRACTOR_CLASSES

        assert "yahoo" in _EXTRACTOR_CLASSES
        assert "alpaca" in _EXTRACTOR_CLASSES
        assert "polygon" in _EXTRACTOR_CLASSES

    @patch("pipeline.extract.prices_daily.get_settings")
    def test_fallback_extractor_warning_on_missing_creds(self, mock_settings):
        from pipeline.extract.prices_daily import PriceExtractor

        mock_cfg = MagicMock()
        mock_cfg.prices.source = "yahoo"
        mock_cfg.prices.fallback_source = "alpaca"
        mock_cfg.prices.universe = ["SPY"]
        mock_cfg.prices.adjust_corporate_actions = True
        mock_cfg.prices.alpaca_api_key = None
        mock_cfg.prices.alpaca_secret_key = None
        mock_cfg.prices.polygon_api_key = None
        mock_settings.return_value = mock_cfg

        import os

        # Clear env vars to ensure ValueError
        with patch.dict(os.environ, {}, clear=True):
            extractor = PriceExtractor()
            # Fallback should be None since credentials are missing
            assert extractor._fallback_extractor is None


# ---------------------------------------------------------------------------
# Alpaca extractor tests
# ---------------------------------------------------------------------------


class TestAlpacaPriceExtractor:
    """Tests for AlpacaPriceExtractor."""

    @patch("pipeline.extract.prices_daily.get_settings")
    def test_alpaca_get_ticker_data(self, mock_settings):
        from pipeline.extract.prices_daily import AlpacaPriceExtractor

        mock_cfg = MagicMock()
        mock_cfg.prices.alpaca_api_key = "test_key"
        mock_cfg.prices.alpaca_secret_key = "test_secret"
        mock_settings.return_value = mock_cfg

        extractor = AlpacaPriceExtractor(api_key="test_key", secret_key="test_secret")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "bars": [
                {"t": "2024-01-02T05:00:00Z", "o": 100, "h": 105, "l": 95, "c": 102, "v": 1000},
                {"t": "2024-01-03T05:00:00Z", "o": 102, "h": 107, "l": 97, "c": 104, "v": 1200},
            ],
            "next_page_token": None,
        }
        mock_response.raise_for_status = MagicMock()
        extractor.client = MagicMock()
        extractor.client.get.return_value = mock_response

        df = extractor.get_ticker_data("AAPL", date(2024, 1, 1), date(2024, 1, 5))

        assert len(df) == 2
        assert "close" in df.columns
        assert df.iloc[0]["close"] == 102
        assert df.iloc[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Polygon extractor tests
# ---------------------------------------------------------------------------


class TestPolygonPriceExtractor:
    """Tests for PolygonPriceExtractor."""

    @patch("pipeline.extract.prices_daily.get_settings")
    def test_polygon_get_ticker_data(self, mock_settings):
        from pipeline.extract.prices_daily import PolygonPriceExtractor

        mock_cfg = MagicMock()
        mock_cfg.prices.polygon_api_key = "test_key"
        mock_settings.return_value = mock_cfg

        extractor = PolygonPriceExtractor(api_key="test_key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"t": 1704153600000, "o": 100, "h": 105, "l": 95, "c": 102, "v": 1000},
                {"t": 1704240000000, "o": 102, "h": 107, "l": 97, "c": 104, "v": 1200},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        extractor.client = MagicMock()
        extractor.client.get.return_value = mock_response

        df = extractor.get_ticker_data("AAPL", date(2024, 1, 1), date(2024, 1, 5))

        assert len(df) == 2
        assert "close" in df.columns
        assert df.iloc[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Walk-forward validation tests
# ---------------------------------------------------------------------------


class TestWalkForwardRunner:
    """Tests for walk-forward strategy validation."""

    def _make_price_data(self, n_days: int = 600, n_tickers: int = 10) -> dict[str, pd.DataFrame]:
        """Generate synthetic price data for testing."""
        rng = np.random.RandomState(42)
        dates = pd.bdate_range("2020-01-01", periods=n_days)

        data = {}
        for i in range(n_tickers):
            ticker = f"TICK{i}"
            returns = rng.normal(0.0005, 0.02, n_days)
            close = 100.0 * np.exp(np.cumsum(returns))
            data[ticker] = pd.DataFrame(
                {
                    "open": close * (1 + rng.normal(0, 0.005, n_days)),
                    "high": close * (1 + abs(rng.normal(0, 0.01, n_days))),
                    "low": close * (1 - abs(rng.normal(0, 0.01, n_days))),
                    "close": close,
                    "volume": rng.randint(100_000, 1_000_000, n_days),
                },
                index=dates,
            )

        return data

    def test_walk_forward_basic(self):
        from pipeline.strategy.strategy_definition import cross_sectional_momentum_strategy
        from pipeline.strategy.walk_forward_runner import (
            WalkForwardConfig,
            run_walk_forward_validation,
        )

        strategy = cross_sectional_momentum_strategy()
        price_data = self._make_price_data(n_days=600, n_tickers=10)

        config = WalkForwardConfig(
            train_days=252,
            test_days=63,
            expanding=True,
            embargo_days=5,
        )

        result = run_walk_forward_validation(strategy, price_data, config)

        assert result.strategy_name == "QSG-SYSTEMATIC-MOM-001"
        assert result.n_folds > 0
        assert len(result.in_sample_metrics) == len(result.out_of_sample_metrics)
        assert len(result.folds) == result.n_folds

    def test_walk_forward_summary(self):
        from pipeline.strategy.strategy_definition import cross_sectional_momentum_strategy
        from pipeline.strategy.walk_forward_runner import (
            WalkForwardConfig,
            run_walk_forward_validation,
        )

        strategy = cross_sectional_momentum_strategy()
        price_data = self._make_price_data(n_days=600, n_tickers=10)

        config = WalkForwardConfig(train_days=252, test_days=63)
        result = run_walk_forward_validation(strategy, price_data, config)

        summary = result.summary()
        assert not summary.empty
        assert "sharpe_ratio" in summary.columns
        assert "total_return" in summary.columns
        # Last row should be MEAN
        assert summary.iloc[-1]["fold"] == "MEAN"

    def test_walk_forward_insufficient_data(self):
        from pipeline.strategy.strategy_definition import cross_sectional_momentum_strategy
        from pipeline.strategy.walk_forward_runner import (
            WalkForwardConfig,
            run_walk_forward_validation,
        )

        strategy = cross_sectional_momentum_strategy()
        # Only 100 days — not enough for walk-forward
        price_data = self._make_price_data(n_days=100, n_tickers=5)

        config = WalkForwardConfig(train_days=252, test_days=63)
        result = run_walk_forward_validation(strategy, price_data, config)

        assert result.n_folds == 0
        assert result.oos_mean_metrics == {}

    def test_walk_forward_metrics_structure(self):
        from pipeline.strategy.walk_forward_runner import _compute_fold_metrics

        # Create a simple return series
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.003])
        metrics = _compute_fold_metrics(returns, cost_bps=5.0)

        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert metrics["max_drawdown"] <= 0  # Drawdowns are negative

    def test_walk_forward_empty_returns(self):
        from pipeline.strategy.walk_forward_runner import _compute_fold_metrics

        metrics = _compute_fold_metrics(pd.Series(dtype=float), cost_bps=5.0)
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Settings tests for new fields
# ---------------------------------------------------------------------------


class TestPriceSettingsNewFields:
    """Tests for new PriceSettings fields."""

    @patch("pipeline.settings.Path.exists", return_value=False)
    @patch("pipeline.settings.Path.mkdir")
    def test_default_fallback_is_none(self, mock_mkdir, mock_exists):
        from pipeline.settings import PriceSettings

        settings = PriceSettings()
        assert settings.fallback_source is None
        assert settings.adjust_corporate_actions is True
        assert settings.alpaca_api_key is None
        assert settings.polygon_api_key is None

    @patch("pipeline.settings.Path.exists", return_value=False)
    @patch("pipeline.settings.Path.mkdir")
    def test_fallback_can_be_set(self, mock_mkdir, mock_exists):
        from pipeline.settings import PriceSettings

        settings = PriceSettings(fallback_source="alpaca")
        assert settings.fallback_source == "alpaca"
