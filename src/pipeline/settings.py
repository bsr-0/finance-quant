"""Configuration and settings management."""

from pathlib import Path

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    api_key: str | None = None
    base_url: str = "https://api.stlouisfed.org/fred"
    series_codes: list[str] = Field(
        default_factory=lambda: ["GDP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "T10Y2Y", "VIXCLS"]
    )
    enabled: bool = True


class GDELTSettings(BaseSettings):
    """GDELT settings."""

    model_config = SettingsConfigDict(env_prefix="GDELT_")

    base_url: str = "https://api.gdeltproject.org/api/v2"
    enabled: bool = True
    max_days_per_request: int = 30


class PolymarketSettings(BaseSettings):
    """Polymarket API settings."""

    model_config = SettingsConfigDict(env_prefix="POLYMARKET_")

    base_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    enabled: bool = True
    rate_limit_per_sec: float = 5.0


class PriceSettings(BaseSettings):
    """Market price data settings."""

    model_config = SettingsConfigDict(env_prefix="PRICES_")

    source: str = "yahoo"  # or "alphavantage", "polygon"
    api_key: str | None = None
    enabled: bool = True
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
    metrics_export_path: Path | None = None

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300


class PipelineSettings(BaseSettings):
    """Main pipeline configuration."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Paths
    raw_lake_path: Path = Path("data/raw")

    # Database
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    # Sources
    fred: FredSettings = Field(default_factory=FredSettings)
    gdelt: GDELTSettings = Field(default_factory=GDELTSettings)
    polymarket: PolymarketSettings = Field(default_factory=PolymarketSettings)
    prices: PriceSettings = Field(default_factory=PriceSettings)

    # Infrastructure
    infrastructure: InfrastructureSettings = Field(default_factory=InfrastructureSettings)

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
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Global settings instance
_settings: PipelineSettings | None = None


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
