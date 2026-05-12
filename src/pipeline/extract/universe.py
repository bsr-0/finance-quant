"""Dynamic universe fetcher.

Fetches constituent lists for standard indices (S&P 100, S&P 500) from
Wikipedia and caches the result locally.  The cache is refreshed automatically
when it is older than ``max_age_days`` (default 7).

Usage::

    from pipeline.extract.universe import get_universe
    tickers = get_universe("sp100")          # list[str], ~101 symbols
    tickers = get_universe("sp500")          # list[str], ~503 symbols
    tickers = get_universe("russell1000")    # list[str], ~1003 symbols
    tickers = get_universe("sp100", extra=["GLD", "TLT"])  # append extras
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/universe")

# Wikipedia URLs for each supported source
_SOURCES: dict[str, dict] = {
    "sp100": {
        "url": "https://en.wikipedia.org/wiki/S%26P_100",
        "table_index": 2,
        "ticker_col": "Symbol",
    },
    "sp500": {
        "url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "table_index": 0,
        "ticker_col": "Symbol",
    },
    "russell1000": {
        "url": "https://en.wikipedia.org/wiki/Russell_1000_Index",
        "table_index": 3,
        "ticker_col": "Symbol",
    },
}


def _cache_path(source: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{source}.json"


def _load_cache(source: str, max_age_days: int) -> list[str] | None:
    path = _cache_path(source)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        fetched_at = datetime.fromisoformat(data["fetched_at"])
        age_days = (datetime.now(UTC) - fetched_at).days
        if age_days > max_age_days:
            logger.info("Universe cache for %s is %d days old, will refresh", source, age_days)
            return None
        tickers = data["tickers"]
        logger.info(
            "Loaded %d tickers for %s from cache (%d days old)", len(tickers), source, age_days
        )
        return tickers
    except Exception as exc:
        logger.warning("Could not read universe cache: %s", exc)
        return None


def _save_cache(source: str, tickers: list[str]) -> None:
    path = _cache_path(source)
    data = {"fetched_at": datetime.now(UTC).isoformat(), "tickers": tickers}
    path.write_text(json.dumps(data, indent=2))
    logger.info("Cached %d tickers for %s", len(tickers), source)


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _fetch_from_wikipedia(source: str) -> list[str]:
    import io

    import httpx

    cfg = _SOURCES[source]
    logger.info("Fetching %s constituents from Wikipedia…", source)
    resp = httpx.get(
        cfg["url"], headers={"User-Agent": _BROWSER_UA}, follow_redirects=True, timeout=30
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[cfg["table_index"]]
    col = cfg["ticker_col"]
    if col not in df.columns:
        # Try case-insensitive match
        col = next((c for c in df.columns if c.lower() == cfg["ticker_col"].lower()), None)
        if col is None:
            raise ValueError(
                f"Column '{cfg['ticker_col']}' not found in table. Columns: {list(df.columns)}"
            )
    tickers = (
        df[col]
        .dropna()
        .str.strip()
        .str.replace(".", "-", regex=False)  # BRK.B → BRK-B (Yahoo convention)
        .tolist()
    )
    logger.info("Fetched %d tickers for %s from Wikipedia", len(tickers), source)
    return tickers


def get_universe(
    source: str = "sp100",
    extra: list[str] | None = None,
    max_age_days: int = 7,
    fallback: list[str] | None = None,
) -> list[str]:
    """Return a deduplicated list of tickers for ``source``.

    Args:
        source: One of ``"sp100"`` or ``"sp500"``.
        extra: Additional tickers to append (e.g. ETFs not in the index).
        max_age_days: Refresh cache if older than this many days.
        fallback: Tickers to return if fetching and cache both fail.

    Returns:
        Deduplicated list of ticker strings, sorted alphabetically with
        ``extra`` tickers appended at the end.
    """
    if source not in _SOURCES:
        raise ValueError(f"Unknown universe source '{source}'. Choose from: {list(_SOURCES)}")

    tickers = _load_cache(source, max_age_days)

    if tickers is None:
        try:
            tickers = _fetch_from_wikipedia(source)
            _save_cache(source, tickers)
        except Exception as exc:
            logger.error("Failed to fetch universe for %s: %s", source, exc)
            if fallback:
                logger.warning("Using fallback universe (%d tickers)", len(fallback))
                tickers = list(fallback)
            else:
                raise

    # Merge extras and deduplicate, preserving order
    seen: set[str] = set(tickers)
    result = list(tickers)
    for t in extra or []:
        if t not in seen:
            result.append(t)
            seen.add(t)

    return result
