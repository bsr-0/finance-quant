"""Universe selection module for systematic equity strategies.

Provides configurable universe definitions with eligibility filters for
tradeable instruments. Supports equities and ETFs with an extensible
interface for futures and options.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import pandas as pd

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    EQUITY = "equity"
    ETF = "etf"
    INDEX_FUTURE = "index_future"
    OPTION = "option"


class Region(Enum):
    US = "US"
    EUROPE = "EU"
    ASIA_PACIFIC = "APAC"
    GLOBAL = "GLOBAL"


class Exchange(Enum):
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    ARCA = "ARCA"
    LSE = "LSE"
    TSE = "TSE"


@dataclass(frozen=True)
class InstrumentMetadata:
    """Metadata for a single tradeable instrument."""

    ticker: str
    name: str = ""
    asset_class: AssetClass = AssetClass.EQUITY
    sector: str = ""
    industry: str = ""
    country: str = "US"
    region: Region = Region.US
    exchange: Exchange = Exchange.NYSE
    market_cap: float = 0.0
    adv_shares: float = 0.0
    adv_dollars: float = 0.0
    avg_spread_bps: float = 0.0
    is_shortable: bool = True


@dataclass
class UniverseFilter:
    """Configurable eligibility filters for universe membership."""

    asset_classes: list[AssetClass] = field(
        default_factory=lambda: [AssetClass.EQUITY, AssetClass.ETF]
    )
    regions: list[Region] = field(default_factory=lambda: [Region.US])
    exchanges: list[Exchange] = field(
        default_factory=lambda: [Exchange.NYSE, Exchange.NASDAQ, Exchange.ARCA]
    )
    min_adv_dollars: float = 1e8  # $100M minimum ADV
    min_price: float = 5.0
    max_price: float = float("inf")
    min_market_cap: float = 0.0
    max_spread_bps: float = 5.0
    require_shortable: bool = False
    exclude_tickers: list[str] = field(default_factory=list)
    include_only_tickers: list[str] | None = None
    exclude_sectors: list[str] = field(default_factory=list)
    max_sector_count: int | None = None
    earnings_blackout_days: int = 3


class UniverseProvider(Protocol):
    """Protocol for data providers that supply instrument metadata."""

    def get_instruments(self) -> list[InstrumentMetadata]:
        """Return all available instruments with metadata."""
        ...

    def get_prices(self, tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        """Return OHLCV DataFrames keyed by ticker."""
        ...


@dataclass
class Universe:
    """A filtered set of tradeable instruments with metadata.

    This is the output of the universe selection process. It can be
    iterated, queried by sector/region, and passed to the signal and
    backtest layers.
    """

    instruments: list[InstrumentMetadata]
    filter_config: UniverseFilter
    as_of_date: str = ""

    @property
    def tickers(self) -> list[str]:
        return [inst.ticker for inst in self.instruments]

    @property
    def ticker_set(self) -> set[str]:
        return {inst.ticker for inst in self.instruments}

    def by_sector(self) -> dict[str, list[InstrumentMetadata]]:
        sectors: dict[str, list[InstrumentMetadata]] = {}
        for inst in self.instruments:
            sectors.setdefault(inst.sector, []).append(inst)
        return sectors

    def by_region(self) -> dict[str, list[InstrumentMetadata]]:
        regions: dict[str, list[InstrumentMetadata]] = {}
        for inst in self.instruments:
            regions.setdefault(inst.region.value, []).append(inst)
        return regions

    def get_instrument(self, ticker: str) -> InstrumentMetadata | None:
        for inst in self.instruments:
            if inst.ticker == ticker:
                return inst
        return None

    def metadata_df(self) -> pd.DataFrame:
        """Return instrument metadata as a DataFrame."""
        records = []
        for inst in self.instruments:
            records.append(
                {
                    "ticker": inst.ticker,
                    "name": inst.name,
                    "asset_class": inst.asset_class.value,
                    "sector": inst.sector,
                    "industry": inst.industry,
                    "country": inst.country,
                    "region": inst.region.value,
                    "exchange": inst.exchange.value,
                    "market_cap": inst.market_cap,
                    "adv_dollars": inst.adv_dollars,
                    "avg_spread_bps": inst.avg_spread_bps,
                }
            )
        return pd.DataFrame(records)

    def __len__(self) -> int:
        return len(self.instruments)


class UniverseBuilder:
    """Build a filtered universe from instrument metadata."""

    def __init__(self, filters: UniverseFilter | None = None) -> None:
        self.filters = filters or UniverseFilter()

    def build(
        self,
        instruments: list[InstrumentMetadata],
        as_of_date: str = "",
    ) -> Universe:
        """Apply all filters and return the eligible universe."""
        f = self.filters
        eligible: list[InstrumentMetadata] = []

        for inst in instruments:
            if not self._passes_filters(inst, f):
                continue
            eligible.append(inst)

        logger.info(
            "Universe built: %d / %d instruments passed filters (as_of=%s)",
            len(eligible),
            len(instruments),
            as_of_date,
        )
        return Universe(
            instruments=eligible,
            filter_config=f,
            as_of_date=as_of_date,
        )

    @staticmethod
    def _passes_filters(inst: InstrumentMetadata, f: UniverseFilter) -> bool:
        if f.include_only_tickers is not None and inst.ticker not in f.include_only_tickers:
            return False

        if inst.ticker in f.exclude_tickers:
            return False

        if inst.asset_class not in f.asset_classes:
            return False

        if inst.region not in f.regions:
            return False

        if inst.exchange not in f.exchanges:
            return False

        if inst.adv_dollars < f.min_adv_dollars:
            return False

        if inst.avg_spread_bps > f.max_spread_bps:
            return False

        if inst.market_cap < f.min_market_cap:
            return False

        if f.require_shortable and not inst.is_shortable:
            return False

        return inst.sector not in f.exclude_sectors

    def build_from_prices(
        self,
        price_data: dict[str, pd.DataFrame],
        metadata: dict[str, dict] | None = None,
        as_of_date: str = "",
    ) -> Universe:
        """Build universe from price DataFrames, inferring metadata where needed."""
        instruments: list[InstrumentMetadata] = []
        for ticker, df in price_data.items():
            if df.empty:
                continue
            meta = (metadata or {}).get(ticker, {})
            close = df["close"]
            volume = df.get("volume", pd.Series(dtype=float))
            last_price = float(close.iloc[-1]) if len(close) > 0 else 0.0

            if not (self.filters.min_price <= last_price <= self.filters.max_price):
                continue

            adv_shares = float(volume.tail(20).mean()) if len(volume) >= 20 else 0.0
            adv_dollars = adv_shares * last_price

            instruments.append(
                InstrumentMetadata(
                    ticker=ticker,
                    name=meta.get("name", ticker),
                    asset_class=AssetClass(meta.get("asset_class", "equity")),
                    sector=meta.get("sector", ""),
                    industry=meta.get("industry", ""),
                    country=meta.get("country", "US"),
                    region=Region(meta.get("region", "US")),
                    exchange=Exchange(meta.get("exchange", "NYSE")),
                    market_cap=meta.get("market_cap", 0.0),
                    adv_shares=adv_shares,
                    adv_dollars=adv_dollars,
                    avg_spread_bps=meta.get("avg_spread_bps", 2.0),
                )
            )

        return self.build(instruments, as_of_date)


# ---------------------------------------------------------------------------
# Pre-built universe definitions
# ---------------------------------------------------------------------------

US_LARGE_CAP_EQUITY = UniverseFilter(
    asset_classes=[AssetClass.EQUITY],
    regions=[Region.US],
    exchanges=[Exchange.NYSE, Exchange.NASDAQ],
    min_adv_dollars=5e8,
    min_price=10.0,
    min_market_cap=10e9,
)

US_ETF_CORE = UniverseFilter(
    asset_classes=[AssetClass.ETF],
    regions=[Region.US],
    exchanges=[Exchange.NYSE, Exchange.NASDAQ, Exchange.ARCA],
    min_adv_dollars=1e8,
    min_price=5.0,
)

US_BROAD_EQUITY = UniverseFilter(
    asset_classes=[AssetClass.EQUITY, AssetClass.ETF],
    regions=[Region.US],
    exchanges=[Exchange.NYSE, Exchange.NASDAQ, Exchange.ARCA],
    min_adv_dollars=1e8,
    min_price=5.0,
)
