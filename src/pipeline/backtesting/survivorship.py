"""Survivorship bias handling and point-in-time universe management.

Ensures backtests use only instruments that were available at each
historical point in time, including delisted names, ticker changes,
and corporate actions.

Components:
    - ``SymbolUniverse``: Manages point-in-time symbol membership,
      including listing/delisting dates and ticker changes.
    - ``CorporateActionMapper``: Maps tickers through renames, mergers,
      and spin-offs to maintain a consistent identifier.
    - ``UniverseFilter``: Filters a backtest universe at each timestamp
      to only include symbols that were live at that time.

Assumptions:
    - Corporate actions data is provided externally (CSV or DataFrame).
    - The universe is reconstructed at each rebalance date, not cached
      across the entire backtest.
    - Delisted symbols have a terminal date; price data after that date
      is excluded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Metadata for a single symbol in the universe.

    Attributes:
        ticker: Current (or final) ticker symbol.
        name: Company/instrument name.
        listing_date: Date the symbol first became tradeable.
        delisting_date: Date the symbol was removed (None if still active).
        delisting_reason: Reason for delisting (e.g. "merged", "bankrupt").
        sector: Sector classification.
        previous_tickers: List of prior tickers for this entity.
    """

    ticker: str
    name: str = ""
    listing_date: pd.Timestamp | None = None
    delisting_date: pd.Timestamp | None = None
    delisting_reason: str = ""
    sector: str = ""
    previous_tickers: list[str] = field(default_factory=list)


@dataclass
class CorporateAction:
    """A single corporate action event.

    Attributes:
        date: Effective date of the action.
        old_ticker: Ticker before the action.
        new_ticker: Ticker after the action (or None for delistings).
        action_type: Type of action (rename, merger, spinoff, delist).
        adjustment_factor: Price adjustment factor (e.g. 2.0 for 2:1 split).
    """

    date: pd.Timestamp
    old_ticker: str
    new_ticker: str | None = None
    action_type: str = "rename"
    adjustment_factor: float = 1.0


class SymbolUniverse:
    """Point-in-time symbol universe management.

    Maintains a registry of all symbols (including dead/delisted ones)
    and provides methods to query which symbols were available at any
    historical date.

    Usage::

        universe = SymbolUniverse()
        universe.add_symbol(SymbolInfo("AAPL", listing_date=pd.Timestamp("1980-12-12")))
        universe.add_symbol(SymbolInfo("ENRN", delisting_date=pd.Timestamp("2001-12-02")))

        # Get symbols available on a specific date
        active = universe.get_active_symbols(pd.Timestamp("2000-01-01"))
        assert "AAPL" in active
        assert "ENRN" in active  # Still active in 2000

        active_2002 = universe.get_active_symbols(pd.Timestamp("2002-01-01"))
        assert "ENRN" not in active_2002  # Delisted
    """

    def __init__(self) -> None:
        self._symbols: dict[str, SymbolInfo] = {}

    def add_symbol(self, info: SymbolInfo) -> None:
        """Register a symbol in the universe."""
        self._symbols[info.ticker] = info

    def add_symbols_from_df(self, df: pd.DataFrame) -> None:
        """Bulk-load symbols from a DataFrame.

        Expected columns: ticker, name (opt), listing_date (opt),
        delisting_date (opt), delisting_reason (opt), sector (opt).
        """
        for _, row in df.iterrows():
            listing = (
                pd.Timestamp(row["listing_date"])
                if "listing_date" in row and pd.notna(row.get("listing_date"))
                else None
            )
            delisting = (
                pd.Timestamp(row["delisting_date"])
                if "delisting_date" in row and pd.notna(row.get("delisting_date"))
                else None
            )
            self.add_symbol(
                SymbolInfo(
                    ticker=str(row["ticker"]),
                    name=str(row.get("name", "")),
                    listing_date=listing,
                    delisting_date=delisting,
                    delisting_reason=str(row.get("delisting_reason", "")),
                    sector=str(row.get("sector", "")),
                )
            )

    def get_active_symbols(self, date: pd.Timestamp) -> list[str]:
        """Return symbols that were active (listed and not yet delisted) on *date*."""
        active = []
        for ticker, info in self._symbols.items():
            if info.listing_date and date < info.listing_date:
                continue
            if info.delisting_date and date > info.delisting_date:
                continue
            active.append(ticker)
        return active

    def get_delisted_before(self, date: pd.Timestamp) -> list[str]:
        """Return symbols that were delisted before *date*."""
        return [
            t
            for t, info in self._symbols.items()
            if info.delisting_date and info.delisting_date < date
        ]

    @property
    def all_symbols(self) -> list[str]:
        return list(self._symbols.keys())

    @property
    def active_count(self) -> int:
        return sum(1 for info in self._symbols.values() if info.delisting_date is None)

    @property
    def delisted_count(self) -> int:
        return sum(1 for info in self._symbols.values() if info.delisting_date is not None)

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self._symbols),
            "active": self.active_count,
            "delisted": self.delisted_count,
        }


class CorporateActionMapper:
    """Map tickers through corporate actions (renames, mergers, splits).

    Maintains a chain of ticker changes so that historical data can
    be aligned with current identifiers.
    """

    def __init__(self) -> None:
        self._actions: list[CorporateAction] = []
        self._forward_map: dict[str, str] = {}  # old → newest
        self._reverse_map: dict[str, list[str]] = {}  # new → [old1, old2]
        self._adjustment_factors: dict[str, float] = {}

    def add_action(self, action: CorporateAction) -> None:
        """Register a corporate action."""
        self._actions.append(action)
        if action.new_ticker:
            self._forward_map[action.old_ticker] = action.new_ticker
            if action.new_ticker not in self._reverse_map:
                self._reverse_map[action.new_ticker] = []
            self._reverse_map[action.new_ticker].append(action.old_ticker)

        if action.adjustment_factor != 1.0:
            self._adjustment_factors[action.old_ticker] = action.adjustment_factor

    def add_actions_from_df(self, df: pd.DataFrame) -> None:
        """Bulk-load corporate actions from a DataFrame.

        Expected columns: date, old_ticker, new_ticker (opt),
        action_type (opt), adjustment_factor (opt).
        """
        for _, row in df.iterrows():
            self.add_action(
                CorporateAction(
                    date=pd.Timestamp(row["date"]),
                    old_ticker=str(row["old_ticker"]),
                    new_ticker=str(row["new_ticker"]) if pd.notna(row.get("new_ticker")) else None,
                    action_type=str(row.get("action_type", "rename")),
                    adjustment_factor=float(row.get("adjustment_factor", 1.0)),
                )
            )

    def resolve_current(self, old_ticker: str) -> str:
        """Follow the chain of renames to find the current ticker."""
        current = old_ticker
        visited = set()
        while current in self._forward_map and current not in visited:
            visited.add(current)
            current = self._forward_map[current]
        return current

    def resolve_historical(self, current_ticker: str) -> list[str]:
        """Find all historical tickers that map to *current_ticker*."""
        result = [current_ticker]
        queue = [current_ticker]
        visited = set()
        while queue:
            ticker = queue.pop(0)
            if ticker in visited:
                continue
            visited.add(ticker)
            for old in self._reverse_map.get(ticker, []):
                result.append(old)
                queue.append(old)
        return result

    def get_price_adjustment(self, ticker: str) -> float:
        """Return the cumulative price adjustment factor for a ticker."""
        factor = 1.0
        current = ticker
        visited = set()
        while current in self._adjustment_factors and current not in visited:
            visited.add(current)
            factor *= self._adjustment_factors[current]
            current = self._forward_map.get(current, current)
        return factor


def filter_universe_at_date(
    price_data: dict[str, pd.DataFrame],
    universe: SymbolUniverse,
    date: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Filter a price data dictionary to only include symbols active at *date*.

    Also truncates each DataFrame to only include data up to *date*
    (preventing look-ahead).
    """
    active = set(universe.get_active_symbols(date))
    filtered = {}
    for sym, df in price_data.items():
        if sym not in active:
            continue
        # Only include data up to the given date
        mask = df.index <= date
        if mask.any():
            filtered[sym] = df.loc[mask]
    return filtered
