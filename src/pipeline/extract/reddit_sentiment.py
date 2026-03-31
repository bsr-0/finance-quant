"""Reddit financial sentiment extractor."""

from __future__ import annotations

import logging
import re
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from pipeline.extract._base import HttpClientMixin
from pipeline.infrastructure.circuit_breaker import get_circuit_breaker
from pipeline.infrastructure.metrics import PipelineMetrics
from pipeline.settings import get_settings

logger = logging.getLogger(__name__)

# Subreddits with high signal for equities
DEFAULT_SUBREDDITS = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
]

# Simple regex to detect stock ticker mentions ($AAPL or AAPL-style)
_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{2,5})(?!\w)")


class RedditSentimentExtractor(HttpClientMixin):
    """Extract posts from financial subreddits for sentiment analysis."""

    def __init__(self) -> None:
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "MarketDataWarehouse/1.0 (research bot)",
            },
        )
        self._circuit = get_circuit_breaker("reddit", failure_threshold=5, recovery_timeout=60.0)
        self._metrics = PipelineMetrics("reddit_sentiment_extractor")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 100,
        after: str | None = None,
    ) -> list[dict]:
        """Fetch posts from a subreddit using Reddit's JSON API."""

        def _do() -> list[dict]:
            url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
            params: dict[str, str | int] = {"limit": limit, "raw_json": 1}
            if after:
                params["after"] = after
            resp = self.client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            children = data.get("data", {}).get("children", [])
            return [c.get("data", {}) for c in children]

        return self._circuit.call(_do)

    @staticmethod
    def _extract_tickers(text: str, known_tickers: set[str] | None = None) -> list[str]:
        """Extract stock ticker mentions from text."""
        if not text:
            return []
        matches = _TICKER_RE.findall(text)
        tickers = set()
        for dollar_match, plain_match in matches:
            t = dollar_match or plain_match
            if t and len(t) >= 2:
                # Filter out common English words
                if t in {
                    "THE",
                    "FOR",
                    "AND",
                    "BUT",
                    "NOT",
                    "ARE",
                    "HAS",
                    "WAS",
                    "CAN",
                    "HIS",
                    "HER",
                    "ALL",
                    "ONE",
                    "TWO",
                    "DAY",
                    "HOW",
                    "NOW",
                    "OLD",
                    "NEW",
                    "BIG",
                    "TOP",
                    "OUT",
                    "OFF",
                    "GET",
                    "GOT",
                    "PUT",
                    "SET",
                    "SAY",
                    "TOO",
                    "USE",
                    "WAY",
                    "MAY",
                    "WHO",
                    "DID",
                    "ITS",
                    "LET",
                    "LOT",
                    "RUN",
                    "TRY",
                    "ASK",
                    "OWN",
                    "WHY",
                    "MEN",
                    "YET",
                    "OUR",
                    "ANY",
                    "FEW",
                    "IMO",
                    "YOLO",
                    "TIL",
                    "PSA",
                    "CEO",
                    "CFO",
                    "IPO",
                    "SEC",
                    "GDP",
                    "FED",
                    "ATH",
                    "EOD",
                    "EPS",
                    "ETF",
                    "ITM",
                    "OTM",
                    "ATM",
                    "RSI",
                    "DCA",
                }:
                    continue
                if known_tickers and t not in known_tickers:
                    continue
                tickers.add(t)
        return sorted(tickers)

    def _parse_posts(
        self,
        posts: list[dict],
        subreddit: str,
        known_tickers: set[str] | None = None,
    ) -> list[dict]:
        """Parse Reddit posts into structured records."""
        rows: list[dict] = []
        for post in posts:
            title = post.get("title", "")
            selftext = post.get("selftext", "")
            full_text = f"{title} {selftext}"
            tickers = self._extract_tickers(full_text, known_tickers)

            created = post.get("created_utc")
            created_dt = datetime.fromtimestamp(created, tz=UTC) if created else datetime.now(UTC)

            rows.append(
                {
                    "post_id": post.get("id", ""),
                    "subreddit": subreddit,
                    "title": title,
                    "selftext": selftext[:5000] if selftext else None,  # Truncate long posts
                    "author": post.get("author"),
                    "score": post.get("score", 0),
                    "upvote_ratio": post.get("upvote_ratio"),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": created_dt,
                    "tickers_mentioned": tickers if tickers else None,
                }
            )

        return rows

    def extract_to_raw(
        self,
        output_dir: Path,
        subreddits: list[str] | None = None,
        run_id: str | None = None,
        posts_per_subreddit: int = 100,
    ) -> list[Path]:
        """Extract Reddit posts for sentiment analysis."""
        subreddits = subreddits or DEFAULT_SUBREDDITS
        output_dir = Path(output_dir) / "reddit_sentiment"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build known tickers set from price universe
        settings = get_settings()
        known_tickers = {t.upper() for t in settings.prices.universe}

        saved_files: list[Path] = []
        today = date.today()

        for subreddit in subreddits:
            logger.info(f"Extracting posts from r/{subreddit}")
            try:
                with self._metrics.time_operation(f"extract_reddit_{subreddit}"):
                    posts = self._fetch_subreddit_posts(
                        subreddit, sort="hot", limit=posts_per_subreddit
                    )
                    rows = self._parse_posts(posts, subreddit, known_tickers)

                if not rows:
                    continue

                df = pd.DataFrame(rows)
                df["extracted_at"] = datetime.now(UTC)
                df["run_id"] = run_id

                file_path = output_dir / f"{subreddit}_{today}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                self._metrics.record_extracted("reddit_sentiment", len(df))
                logger.info(f"Saved {len(df)} posts from r/{subreddit}")

            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed r/{subreddit}: {e}")
                continue

            time.sleep(2.0)  # Reddit rate limit

        return saved_files


def extract_reddit_sentiment(
    output_dir: Path,
    subreddits: list[str] | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """CLI-friendly wrapper."""
    extractor = RedditSentimentExtractor()
    return extractor.extract_to_raw(
        output_dir=output_dir,
        subreddits=subreddits,
        run_id=run_id,
    )
