"""Unit tests for data extractors (no database required)."""

import io
import zipfile
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


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

    @patch("pipeline.extract.prices_daily.get_settings")
    def test_get_ticker_data_missing_quotes(self, mock_settings):
        """Response with timestamps but no quote data returns empty DataFrame."""
        mock_settings.return_value.prices.source = "yahoo"
        mock_settings.return_value.prices.universe = ["SPY"]

        from pipeline.extract.prices_daily import YahooFinanceExtractor

        extractor = YahooFinanceExtractor()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "chart": {"result": [{"timestamp": [1704067200], "indicators": {}}]}
        }
        mock_response.raise_for_status = MagicMock()
        extractor.client = MagicMock()
        extractor.client.get.return_value = mock_response

        df = extractor.get_ticker_data("SPY", date(2024, 1, 1), date(2024, 1, 2))

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


class TestSecFundamentalsExtractor:
    """Tests for SEC EDGAR fundamentals extractor."""

    def test_parse_facts_extracts_metrics(self):
        from pipeline.extract.sec_fundamentals import SecFundamentalsExtractor

        facts_json = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenue",
                        "units": {
                            "USD": [
                                {
                                    "form": "10-Q",
                                    "filed": "2024-01-15",
                                    "end": "2023-12-31",
                                    "val": 50000000,
                                    "accn": "0000789456-24-001234",
                                    "fy": 2024,
                                    "fp": "Q1",
                                }
                            ]
                        },
                    },
                    "NetIncomeLoss": {
                        "label": "Net Income",
                        "units": {
                            "USD": [
                                {
                                    "form": "10-K",
                                    "filed": "2024-02-28",
                                    "end": "2023-12-31",
                                    "val": 10000000,
                                    "accn": "0000789456-24-005678",
                                    "fy": 2024,
                                    "fp": "FY",
                                }
                            ]
                        },
                    },
                }
            }
        }

        rows = SecFundamentalsExtractor._parse_facts(
            facts_json, ticker="AAPL", cik=320193, metrics=["Revenues", "NetIncomeLoss"]
        )

        assert len(rows) == 2
        revenue_row = [r for r in rows if r["metric_name"] == "Revenues"][0]
        assert revenue_row["ticker"] == "AAPL"
        assert revenue_row["metric_value"] == 50000000.0
        assert revenue_row["form_type"] == "10-Q"
        assert revenue_row["fiscal_period_end"] == "2023-12-31"
        assert revenue_row["filing_date"] == "2024-01-15"

    def test_parse_facts_filters_non_10q_10k(self):
        from pipeline.extract.sec_fundamentals import SecFundamentalsExtractor

        facts_json = {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "label": "Revenue",
                        "units": {
                            "USD": [
                                {
                                    "form": "8-K",
                                    "filed": "2024-01-01",
                                    "end": "2023-12-31",
                                    "val": 100,
                                },
                                {
                                    "form": "10-Q",
                                    "filed": "2024-01-15",
                                    "end": "2023-12-31",
                                    "val": 200,
                                    "accn": "x",
                                    "fy": 2024,
                                    "fp": "Q1",
                                },
                            ]
                        },
                    }
                }
            }
        }

        rows = SecFundamentalsExtractor._parse_facts(
            facts_json,
            ticker="AAPL",
            cik=320193,
            metrics=["Revenues"],
        )
        assert len(rows) == 1
        assert rows[0]["form_type"] == "10-Q"

    def test_parse_facts_empty(self):
        from pipeline.extract.sec_fundamentals import SecFundamentalsExtractor

        rows = SecFundamentalsExtractor._parse_facts({"facts": {}}, ticker="AAPL", cik=320193)
        assert rows == []


class TestSecInsiderExtractor:
    """Tests for SEC EDGAR insider trades extractor."""

    def test_parse_form4_xml_extracts_transactions(self):
        from pipeline.extract.sec_insider import SecInsiderExtractor

        xml_text = """<?xml version="1.0"?>
        <ownershipDocument>
            <reportingOwner>
                <reportingOwnerId>
                    <rptOwnerName>John Doe</rptOwnerName>
                    <rptOwnerCik>1234567</rptOwnerCik>
                </reportingOwnerId>
                <reportingOwnerRelationship>
                    <officerTitle>CEO</officerTitle>
                </reportingOwnerRelationship>
            </reportingOwner>
            <nonDerivativeTransaction>
                <transactionDate><value>2024-01-15</value></transactionDate>
                <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
                <transactionAmounts>
                    <transactionShares><value>1000</value></transactionShares>
                    <transactionPricePerShare><value>75.50</value></transactionPricePerShare>
                </transactionAmounts>
                <postTransactionAmounts>
                    <sharesOwnedFollowingTransaction><value>50000</value></sharesOwnedFollowingTransaction>
                </postTransactionAmounts>
                <ownershipNature>
                    <directOrIndirectOwnership><value>D</value></directOrIndirectOwnership>
                </ownershipNature>
            </nonDerivativeTransaction>
        </ownershipDocument>
        """

        rows = SecInsiderExtractor._parse_form4_xml(
            xml_text,
            ticker="AAPL",
            cik=320193,
            filing_date="2024-01-16",
        )

        assert len(rows) == 1
        assert rows[0]["insider_name"] == "John Doe"
        assert rows[0]["insider_title"] == "CEO"
        assert rows[0]["transaction_type"] == "purchase"
        assert rows[0]["ticker"] == "AAPL"

    def test_parse_form4_xml_invalid(self):
        from pipeline.extract.sec_insider import SecInsiderExtractor

        rows = SecInsiderExtractor._parse_form4_xml(
            "not xml",
            ticker="AAPL",
            cik=320193,
            filing_date="2024-01-16",
        )
        assert rows == []


class TestSec13FExtractor:
    """Tests for SEC 13F institutional holdings extractor."""

    def test_parse_13f_xml_extracts_holdings(self):
        from pipeline.extract.sec_13f import Sec13FExtractor

        xml_text = """<?xml version="1.0"?>
        <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
            <infoTable>
                <nameOfIssuer>Apple Inc</nameOfIssuer>
                <cusip>037833100</cusip>
                <titleOfClass>Common Stock</titleOfClass>
                <value>500000</value>
                <investmentDiscretion>Sole</investmentDiscretion>
                <shrsOrPrnAmt>
                    <sshPrnamt>1000000</sshPrnamt>
                    <sshPrnamtType>SH</sshPrnamtType>
                </shrsOrPrnAmt>
                <votingAuthority>
                    <sole>500000</sole>
                    <shared>500000</shared>
                    <none>0</none>
                </votingAuthority>
            </infoTable>
        </informationTable>
        """

        rows = Sec13FExtractor._parse_13f_xml(
            xml_text,
            filer_cik=1234,
            filer_name="Test Fund",
            report_date="2024-03-31",
            filing_date="2024-05-15",
        )

        assert len(rows) == 1
        assert rows[0]["cusip"] == "037833100"
        assert rows[0]["issuer_name"] == "Apple Inc"
        assert rows[0]["shares_held"] == 1000000
        assert rows[0]["filer_name"] == "Test Fund"

    def test_parse_13f_xml_empty(self):
        from pipeline.extract.sec_13f import Sec13FExtractor

        xml_text = """<?xml version="1.0"?>
        <informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
        </informationTable>
        """

        rows = Sec13FExtractor._parse_13f_xml(
            xml_text,
            filer_cik=1234,
            filer_name="Test Fund",
            report_date="2024-03-31",
            filing_date="2024-05-15",
        )
        assert rows == []


class TestOptionsExtractor:
    """Tests for options chain extractor."""

    def test_parse_chain_extracts_contracts(self):
        from pipeline.extract.options_data import OptionsDataExtractor

        data = {
            "optionChain": {
                "result": [
                    {
                        "options": [
                            {
                                "expirationDate": 1706140800,
                                "calls": [
                                    {
                                        "strike": 190.0,
                                        "lastPrice": 5.50,
                                        "bid": 5.40,
                                        "ask": 5.60,
                                        "volume": 1500,
                                        "openInterest": 3000,
                                        "impliedVolatility": 0.25,
                                        "inTheMoney": True,
                                    }
                                ],
                                "puts": [
                                    {
                                        "strike": 190.0,
                                        "lastPrice": 3.20,
                                        "bid": 3.10,
                                        "ask": 3.30,
                                        "volume": 800,
                                        "openInterest": 2000,
                                        "impliedVolatility": 0.28,
                                        "inTheMoney": False,
                                    }
                                ],
                            }
                        ]
                    }
                ]
            }
        }

        rows = OptionsDataExtractor._parse_chain(None, data, "AAPL", date(2024, 1, 15))

        assert len(rows) == 2
        call_row = [r for r in rows if r["option_type"] == "call"][0]
        assert call_row["ticker"] == "AAPL"
        assert call_row["strike"] == 190.0
        assert call_row["implied_volatility"] == 0.25

        put_row = [r for r in rows if r["option_type"] == "put"][0]
        assert put_row["implied_volatility"] == 0.28

    def test_parse_chain_empty(self):
        from pipeline.extract.options_data import OptionsDataExtractor

        data = {"optionChain": {"result": []}}
        rows = OptionsDataExtractor._parse_chain(None, data, "AAPL", date(2024, 1, 15))
        assert rows == []


class TestEarningsExtractor:
    """Tests for earnings extractor."""

    def test_parse_earnings_data_with_revenue(self):
        from pipeline.extract.earnings import EarningsExtractor

        modules = {
            "earningsHistory": {
                "history": [
                    {
                        "quarter": {"raw": 1696118400, "fmt": "2023-10-01"},
                        "period": "3Q2023",
                        "epsEstimate": {"raw": 1.39},
                        "epsActual": {"raw": 1.46},
                        "epsDifference": {"raw": 0.07},
                        "surprisePercent": {"raw": 0.0504},
                    }
                ]
            },
            "earnings": {
                "financialsChart": {
                    "quarterly": [
                        {
                            "date": "3Q2023",
                            "revenue": {"raw": 89500000000},
                            "earnings": {"raw": 22960000000},
                        },
                    ]
                }
            },
        }

        rows = EarningsExtractor._parse_earnings_data(modules, "AAPL")

        assert len(rows) == 1
        assert rows[0]["eps_actual"] == 1.46
        assert rows[0]["eps_estimate"] == 1.39
        assert rows[0]["eps_surprise_pct"] == 5.04
        assert rows[0]["revenue_actual"] == 89500000000

    def test_parse_earnings_data_empty(self):
        from pipeline.extract.earnings import EarningsExtractor

        rows = EarningsExtractor._parse_earnings_data({}, "AAPL")
        assert rows == []


class TestSettingsDefaults:
    """Tests for expanded settings defaults."""

    @patch.dict("os.environ", {}, clear=False)
    def test_fred_series_codes_expanded(self):
        from pipeline.settings import FredSettings

        settings = FredSettings()
        # Should have 27 series (including FX, credit spreads, housing, etc.)
        assert len(settings.series_codes) >= 25
        # Verify key additions are present
        assert "BAMLH0A0HYM2" in settings.series_codes  # HY spread
        assert "DEXUSEU" in settings.series_codes  # USD/EUR
        assert "NFCI" in settings.series_codes  # Financial conditions
        assert "ICSA" in settings.series_codes  # Initial claims

    @patch.dict("os.environ", {}, clear=False)
    def test_sec_edgar_cusip_mapping(self):
        from pipeline.settings import SecEdgarSettings

        settings = SecEdgarSettings()
        assert len(settings.cusip_mapping) > 0
        assert settings.cusip_mapping.get("AAPL") == "037833100"
        assert settings.cusip_mapping.get("MSFT") == "594918104"


class TestFactorsFFDefensiveGuards:
    """Tests for factors_ff.py defensive guards."""

    def test_read_zip_csv_empty_zip(self):
        """Empty ZIP raises ValueError."""
        from pipeline.extract.factors_ff import _read_zip_csv

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass  # empty ZIP

        with pytest.raises(ValueError, match="Empty ZIP file"):
            _read_zip_csv(buf.getvalue())


class TestGDELTDefensiveGuards:
    """Tests for gdelt.py empty ZIP guard."""

    def test_empty_zip_returns_none(self):
        from pipeline.extract.gdelt import GDELTExtractor

        extractor = GDELTExtractor()

        # Create empty ZIP
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w"):
            pass

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = buf.getvalue()
        mock_response.raise_for_status = MagicMock()
        extractor.client = MagicMock()
        extractor.client.get.return_value = mock_response

        result = extractor.download_day(date(2024, 1, 15))
        assert result is None


class TestShortInterestDefensiveGuards:
    """Tests for short_interest.py type guard on settlement date."""

    @patch("pipeline.extract.short_interest.get_settings")
    def test_non_numeric_short_date_uses_today(self, mock_settings):
        mock_settings.return_value.short_interest.base_url = "https://test.com"

        from pipeline.extract.short_interest import ShortInterestExtractor

        extractor = ShortInterestExtractor()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "quoteSummary": {
                "result": [
                    {
                        "defaultKeyStatistics": {
                            "sharesShort": {"raw": 1000000},
                            "dateShortInterest": {"raw": "not-a-number"},
                            "averageDailyVolume10Day": {"raw": 5000000},
                            "floatShares": {"raw": 50000000},
                            "shortRatio": {"raw": 2.5},
                            "shortPercentOfFloat": {"raw": 0.02},
                        }
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()
        extractor.client = MagicMock()
        extractor.client.get.return_value = mock_response

        rows = extractor._fetch_short_interest("AAPL")

        assert len(rows) == 1
        assert rows[0]["settlement_date"] == date.today()
