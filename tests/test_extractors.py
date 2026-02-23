"""Unit tests for data extractors (no database required)."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd


class TestFredExtractor:
    """Tests for FRED extractor."""

    @patch("pipeline.extract.fred.get_settings")
    def test_get_observations_parses_response(self, mock_settings):
        mock_settings.return_value.fred.api_key = "test_key"
        mock_settings.return_value.fred.base_url = "https://api.test.com"
        mock_settings.return_value.fred.series_codes = ["GDP"]

        from pipeline.extract.fred import FredExtractor

        extractor = FredExtractor(api_key="test_key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {
                    "date": "2024-01-01",
                    "value": "100.5",
                    "realtime_start": "2024-01-01",
                    "realtime_end": "2024-01-01",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        extractor.client = MagicMock()
        extractor.client.get.return_value = mock_response

        df = extractor.get_observations("GDP")

        assert len(df) == 1
        assert df.iloc[0]["series_code"] == "GDP"
        assert df.iloc[0]["value"] == 100.5

    @patch("pipeline.extract.fred.get_settings")
    def test_get_observations_empty_response(self, mock_settings):
        mock_settings.return_value.fred.api_key = "test_key"
        mock_settings.return_value.fred.base_url = "https://api.test.com"
        mock_settings.return_value.fred.series_codes = ["GDP"]

        from pipeline.extract.fred import FredExtractor

        extractor = FredExtractor(api_key="test_key")

        mock_response = MagicMock()
        mock_response.json.return_value = {"observations": []}
        mock_response.raise_for_status = MagicMock()
        extractor.client = MagicMock()
        extractor.client.get.return_value = mock_response

        df = extractor.get_observations("NONEXISTENT")

        assert df.empty

    @patch("pipeline.extract.fred.get_settings")
    def test_extract_to_raw_saves_parquet(self, mock_settings, tmp_path):
        mock_settings.return_value.fred.api_key = "test_key"
        mock_settings.return_value.fred.base_url = "https://api.test.com"
        mock_settings.return_value.fred.series_codes = ["GDP"]

        from pipeline.extract.fred import FredExtractor

        extractor = FredExtractor(api_key="test_key")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "observations": [
                {
                    "date": "2024-01-01",
                    "value": "100.5",
                    "realtime_start": "2024-01-01",
                    "realtime_end": "2024-01-01",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        extractor.client = MagicMock()
        extractor.client.get.return_value = mock_response

        files = extractor.extract_to_raw(
            output_dir=tmp_path,
            series_codes=["GDP"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

        assert len(files) == 1
        assert files[0].suffix == ".parquet"
        df = pd.read_parquet(files[0])
        assert "extracted_at" in df.columns
        assert "series_code" in df.columns


class TestYahooFinanceExtractor:
    """Tests for Yahoo Finance extractor."""

    @patch("pipeline.extract.prices_daily.get_settings")
    def test_get_ticker_data_parses_response(self, mock_settings):
        mock_settings.return_value.prices.source = "yahoo"
        mock_settings.return_value.prices.universe = ["SPY"]

        from pipeline.extract.prices_daily import YahooFinanceExtractor

        extractor = YahooFinanceExtractor()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "chart": {
                "result": [
                    {
                        "timestamp": [1704067200],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [470.0],
                                    "high": [475.0],
                                    "low": [469.0],
                                    "close": [473.0],
                                    "volume": [1000000],
                                }
                            ],
                            "adjclose": [{"adjclose": [473.0]}],
                        },
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        extractor.client = MagicMock()
        extractor.client.get.return_value = mock_response

        df = extractor.get_ticker_data("SPY", date(2024, 1, 1), date(2024, 1, 2))

        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "SPY"
        assert df.iloc[0]["close"] == 473.0

    @patch("pipeline.extract.prices_daily.get_settings")
    def test_get_ticker_data_no_data(self, mock_settings):
        mock_settings.return_value.prices.source = "yahoo"
        mock_settings.return_value.prices.universe = ["SPY"]

        from pipeline.extract.prices_daily import YahooFinanceExtractor

        extractor = YahooFinanceExtractor()

        mock_response = MagicMock()
        mock_response.json.return_value = {"chart": {"result": [{}]}}
        mock_response.raise_for_status = MagicMock()
        extractor.client = MagicMock()
        extractor.client.get.return_value = mock_response

        df = extractor.get_ticker_data("INVALID", date(2024, 1, 1), date(2024, 1, 2))

        assert df.empty


class TestPolymarketExtractor:
    """Tests for Polymarket extractor."""

    @patch("pipeline.extract.polymarket.get_settings")
    def test_get_markets_returns_list(self, mock_settings):
        mock_settings.return_value.polymarket.base_url = "https://clob.test.com"
        mock_settings.return_value.polymarket.gamma_url = "https://gamma.test.com"
        mock_settings.return_value.polymarket.rate_limit_per_sec = 100.0

        from pipeline.extract.polymarket import PolymarketExtractor

        extractor = PolymarketExtractor()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "markets": [
                {"id": "market1", "question": "Will X happen?"},
                {"id": "market2", "question": "Will Y happen?"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        extractor.client = MagicMock()
        extractor.client.get.return_value = mock_response

        markets = extractor.get_markets(limit=10)

        assert len(markets) == 2
        assert markets[0]["id"] == "market1"

    @patch("pipeline.extract.polymarket.get_settings")
    def test_rate_limiting(self, mock_settings):
        mock_settings.return_value.polymarket.base_url = "https://clob.test.com"
        mock_settings.return_value.polymarket.gamma_url = "https://gamma.test.com"
        mock_settings.return_value.polymarket.rate_limit_per_sec = 100.0

        from pipeline.extract.polymarket import PolymarketExtractor

        extractor = PolymarketExtractor()

        # First call sets _last_request_time
        extractor._rate_limit()
        assert extractor._last_request_time is not None
