"""Configuration and settings management."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field
from pydantic import validator as _v1_validator

try:
    from pydantic import field_validator
except ImportError:  # pragma: no cover - pydantic v1 fallback
    field_validator = _v1_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:  # pragma: no cover - pydantic v1 fallback
    from pydantic import BaseSettings  # type: ignore

    SettingsConfigDict = dict  # type: ignore


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    name: str = "market_data"
    user: str = "postgres"
    password: str = "postgres"

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class FredSettings(BaseSettings):
    """FRED API settings."""

    model_config = SettingsConfigDict(env_prefix="FRED_")

    api_key: Optional[str] = None
    base_url: str = "https://api.stlouisfed.org/fred"
    series_codes: list[str] = Field(
        default_factory=lambda: [
            # Core macro indicators
            "GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "T10Y2Y", "VIXCLS",
            "DGS10", "DGS2", "TB3MS",
            # Credit spreads
            "BAMLH0A0HYM2", "BAMLC0A4CBBB",
            # Housing
            "HOUST", "CSUSHPISA",
            # Labor market
            "ICSA", "PAYEMS",
            # Money supply
            "M2SL",
            # Commodities
            "DCOILWTICO", "GOLDAMGBD228NLBM",
            # Dollar index
            "DTWEXBGS",
            # Financial conditions
            "NFCI",
            # Leading indicators
            "USSLIND",
            # Inflation expectations
            "T5YIE", "T10YIE",
            # FX rates (major pairs)
            "DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXCHUS", "DEXCAUS",
        ]
    )
    enabled: bool = True


class GDELTSettings(BaseSettings):
    """GDELT settings."""

    model_config = SettingsConfigDict(env_prefix="GDELT_")

    base_url: str = "https://api.gdeltproject.org/api/v2"
    enabled: bool = True
    max_days_per_request: int = 30
    available_time_source: str = "DATEADDED"


class PolymarketSettings(BaseSettings):
    """Polymarket API settings."""

    model_config = SettingsConfigDict(env_prefix="POLYMARKET_")

    base_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    enabled: bool = True
    rate_limit_per_sec: float = 5.0
    universe_mode: str = "top_volume"  # top_volume | all_active | explicit_list
    top_volume_limit: int = 50
    explicit_markets: list[str] = Field(default_factory=list)
    orderbook_snapshot_freq: str = "5m"
    trades_page_size: int = 100
    trades_max_pages: int = 200


class PriceSettings(BaseSettings):
    """Market price data settings."""

    model_config = SettingsConfigDict(env_prefix="PRICES_")

    source: str = "yahoo"  # or "alphavantage", "polygon"
    api_key: Optional[str] = None
    enabled: bool = True
    market_close_time: str = "16:00:00"
    exchange_timezone: str = "America/New_York"
    vendor_delay_minutes: int = 15
    universe: list[str] = Field(
        default_factory=lambda: [
            "SPY",
            "QQQ",
            "IWM",
            "VTI",
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
        ]
    )


class SecEdgarSettings(BaseSettings):
    """SEC EDGAR data settings."""

    model_config = SettingsConfigDict(env_prefix="SEC_")

    enabled: bool = True
    rate_limit_delay: float = 0.12  # SEC asks for <=10 req/s
    cusip_mapping: dict[str, str] = Field(default_factory=lambda: {
        "AAPL": "037833100", "MSFT": "594918104", "GOOGL": "02079K305",
        "AMZN": "023135106", "META": "30303M102", "TSLA": "88160R101",
        "NVDA": "67066G104", "JPM": "46625H100", "JNJ": "478160104",
        "V": "92826C839", "WMT": "931142103", "PG": "742718109",
        "UNH": "91324P102", "HD": "437076102", "MA": "57636Q104",
    })
    fundamentals_metrics: list[str] = Field(default_factory=lambda: [
        "Revenues",
        "NetIncomeLoss",
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "LongTermDebt",
        "CashAndCashEquivalentsAtCarryingValue",
        "OperatingIncomeLoss",
        "GrossProfit",
        "NetCashProvidedByUsedInOperatingActivities",
        "CommonStockSharesOutstanding",
    ])


class OptionsSettings(BaseSettings):
    """Options data settings."""

    model_config = SettingsConfigDict(env_prefix="OPTIONS_")

    enabled: bool = True
    source: str = "yahoo"
    max_expirations: int = 6


class EarningsSettings(BaseSettings):
    """Earnings calendar settings."""

    model_config = SettingsConfigDict(env_prefix="EARNINGS_")

    enabled: bool = True
    source: str = "yahoo"


class SentimentSettings(BaseSettings):
    """Social media sentiment settings."""

    model_config = SettingsConfigDict(env_prefix="SENTIMENT_")

    enabled: bool = True
    subreddits: list[str] = Field(
        default_factory=lambda: ["wallstreetbets", "stocks", "investing", "options"]
    )
    posts_per_subreddit: int = 100


class ShortInterestSettings(BaseSettings):
    """Short interest settings."""

    model_config = SettingsConfigDict(env_prefix="SHORT_INTEREST_")

    enabled: bool = True


class EtfFlowsSettings(BaseSettings):
    """ETF fund flows settings."""

    model_config = SettingsConfigDict(env_prefix="ETF_FLOWS_")

    enabled: bool = True
    etf_universe: list[str] = Field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "VTI", "VOO",
        "XLF", "XLK", "XLE", "XLV", "XLI",
        "TLT", "IEF", "SHY", "LQD", "HYG",
        "GLD", "SLV", "USO",
        "EEM", "EFA", "VWO",
    ])


class InfrastructureSettings(BaseSettings):
    """Infrastructure and performance settings."""

    model_config = SettingsConfigDict(env_prefix="INFRA_")

    # Async processing
    max_async_workers: int = 10

    # Circuit breaker
    circuit_failure_threshold: int = 5
    circuit_recovery_timeout: float = 30.0

    # Checkpointing
    checkpoint_dir: Path = Path("data/checkpoints")

    # Batch processing
    batch_size: int = 1000
    flush_interval: int = 10000

    # Snapshot building
    snapshot_max_workers: int = 4
    snapshot_batch_size: int = 100

    # Metrics
    metrics_enabled: bool = True
    metrics_export_path: Optional[Path] = None

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    lineage_enabled: bool = True


class FactorSettings(BaseSettings):
    """Factor data settings."""

    model_config = SettingsConfigDict(env_prefix="FACTORS_")

    source: str = "ken_french"
    min_history_days: int = 252
    enabled: bool = True
    available_time_lag_days: int = 1
    available_time_release_time: str = "09:30"
    exchange_timezone: str = "America/New_York"


class EvaluationSettings(BaseSettings):
    """Evaluation settings."""

    model_config = SettingsConfigDict(env_prefix="EVAL_")

    cost_bps: float = 20.0
    edge_threshold: float = 0.02
    rebalance_freq: str = "1d"
    benchmark_symbol: str = "SPY"
    max_leverage: float = 2.0
    max_adv_pct: float = 0.1
    borrow_cost_bps: float = 30.0
    slippage_bps: float = 2.0
    pm_fee_bps: float = 10.0


class HistoricalFixesSettings(BaseSettings):
    """Statistical corrections for historical data limitations."""

    model_config = SettingsConfigDict(env_prefix="HIST_")

    # Latency estimation
    max_backfill_days: int = 5
    min_latency_samples: int = 500
    latency_percentile: float = 0.95
    latency_stats_max_age_hours: int = 24

    # Conservative release timing for macro vintages
    macro_release_time: str = "12:00:00"
    macro_release_timezone: str = "America/New_York"
    macro_release_jitter_minutes: int = 60

    # Outlier detection
    price_outlier_mad_threshold: float = 3.5
    trade_outlier_mad_threshold: float = 4.0

    # Fallback lags (minutes) if latency stats are unavailable
    gdelt_fallback_lag_minutes: int = 180
    polymarket_fallback_lag_minutes: int = 2

    # Universe audit + selection weighting
    polymarket_universe_audit: bool = True
    selection_weight_mode: str = "count"  # count | volume


class PipelineSettings(BaseSettings):
    """Main pipeline configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Paths
    raw_lake_path: Path = Path("data/raw")

    # Database
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    # Sources — original
    fred: FredSettings = Field(default_factory=FredSettings)
    gdelt: GDELTSettings = Field(default_factory=GDELTSettings)
    polymarket: PolymarketSettings = Field(default_factory=PolymarketSettings)
    prices: PriceSettings = Field(default_factory=PriceSettings)

    # Sources — new data sources
    sec_edgar: SecEdgarSettings = Field(default_factory=SecEdgarSettings)
    options: OptionsSettings = Field(default_factory=OptionsSettings)
    earnings: EarningsSettings = Field(default_factory=EarningsSettings)
    sentiment: SentimentSettings = Field(default_factory=SentimentSettings)
    short_interest: ShortInterestSettings = Field(default_factory=ShortInterestSettings)
    etf_flows: EtfFlowsSettings = Field(default_factory=EtfFlowsSettings)

    # Infrastructure
    infrastructure: InfrastructureSettings = Field(default_factory=InfrastructureSettings)

    # Factors & evaluation
    factors: FactorSettings = Field(default_factory=FactorSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    historical_fixes: HistoricalFixesSettings = Field(default_factory=HistoricalFixesSettings)

    # Pipeline behavior
    default_start_date: str = "2020-01-01"
    default_end_date: str = "2024-12-31"
    batch_size: int = 1000

    @field_validator("raw_lake_path")
    @classmethod
    def validate_raw_lake_path(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "PipelineSettings":
        """Load settings from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def to_yaml(self, path: Path) -> None:
        """Save settings to YAML file."""
        with open(path, "w") as f:
            data = self.model_dump() if hasattr(self, "model_dump") else self.dict()
            yaml.dump(data, f, default_flow_style=False)


# Global settings instance
_settings: Optional[PipelineSettings] = None


def get_settings() -> PipelineSettings:
    """Get or create global settings instance."""
    global _settings
    if _settings is None:
        config_path = Path("config.yaml")
        if config_path.exists():
            _settings = PipelineSettings.from_yaml(config_path)
        else:
            _settings = PipelineSettings()
    return _settings


def reset_settings() -> None:
    """Reset global settings (useful for testing)."""
    global _settings
    _settings = None
