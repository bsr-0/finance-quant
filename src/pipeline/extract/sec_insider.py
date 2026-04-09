"""SEC EDGAR Form 4 insider trading extractor."""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
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

SEC_USER_AGENT = "MarketDataWarehouse/1.0 (research; contact@example.com)"

# Map EDGAR transaction codes to human-readable types.
_TXN_CODES = {
    "P": "purchase",
    "S": "sale",
    "A": "grant",
    "D": "disposition",
    "F": "tax",
    "M": "exercise",
    "G": "gift",
    "C": "conversion",
}


class SecInsiderExtractor(HttpClientMixin):
    """Extract insider trading data from SEC EDGAR Form 4 filings."""

    def __init__(self) -> None:
        self.client = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": SEC_USER_AGENT},
            follow_redirects=True,
        )
        self._circuit = get_circuit_breaker(
            "sec_edgar_insider", failure_threshold=5, recovery_timeout=60.0
        )
        self._metrics = PipelineMetrics("sec_insider_extractor")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_recent_filings(self, cik: int, form_type: str = "4", count: int = 40) -> list[dict]:
        """Fetch recent Form 4 filings for a company CIK from EDGAR."""

        def _do() -> list[dict]:
            padded = str(cik).zfill(10)
            url = (
                f"https://efts.sec.gov/LATEST/search-index"
                f"?q=%22{padded}%22&forms={form_type}"
                f"&dateRange=custom&startdt=2020-01-01"
                f"&enddt={date.today().isoformat()}"
            )
            resp = self.client.get(url)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
            return data.get("hits", {}).get("hits", [])

        return self._circuit.call(_do)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_filing_index(self, cik: int, start_date: date | None = None) -> list[dict]:
        """Fetch Form 4 filing list from EDGAR submissions endpoint."""

        def _do() -> list[dict]:
            padded = str(cik).zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{padded}.json"
            resp = self.client.get(url)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()

            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            results = []
            for i, form in enumerate(forms):
                if form != "4":
                    continue
                filing_date = dates[i] if i < len(dates) else None
                if start_date and filing_date and filing_date < start_date.isoformat():
                    continue
                results.append(
                    {
                        "accession_number": accessions[i] if i < len(accessions) else None,
                        "filing_date": filing_date,
                        "primary_document": primary_docs[i] if i < len(primary_docs) else None,
                    }
                )
            return results

        return self._circuit.call(_do)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _fetch_form4_xml(self, cik: int, accession: str, primary_doc: str) -> str | None:
        """Fetch and return raw XML of a Form 4 filing."""

        def _do() -> str | None:
            padded = str(cik).zfill(10)
            acc_clean = accession.replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{padded}/{acc_clean}/{primary_doc}"
            resp = self.client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.text

        return self._circuit.call(_do)

    @staticmethod
    def _parse_form4_xml(xml_text: str, ticker: str, cik: int, filing_date: str) -> list[dict]:
        """Parse Form 4 XML into transaction records."""
        rows: list[dict] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return rows

        # Reporting owner info
        owner_el = root.find(".//reportingOwner")
        insider_name = ""
        insider_title = ""
        insider_cik = None
        if owner_el is not None:
            rid = owner_el.find("reportingOwnerId")
            if rid is not None:
                name_el = rid.find("rptOwnerName")
                insider_name = name_el.text.strip() if name_el is not None and name_el.text else ""
                cik_el = rid.find("rptOwnerCik")
                insider_cik = (
                    int(cik_el.text.strip()) if cik_el is not None and cik_el.text else None
                )
            rel = owner_el.find("reportingOwnerRelationship")
            if rel is not None:
                title_el = rel.find("officerTitle")
                insider_title = (
                    title_el.text.strip() if title_el is not None and title_el.text else ""
                )

        # Non-derivative transactions
        for txn in root.findall(".//nonDerivativeTransaction"):
            row = SecInsiderExtractor._parse_transaction(
                txn, ticker, cik, insider_name, insider_title, insider_cik, filing_date
            )
            if row:
                rows.append(row)

        return rows

    @staticmethod
    def _parse_transaction(
        txn_el, ticker, cik, insider_name, insider_title, insider_cik, filing_date
    ) -> dict | None:
        """Parse a single transaction element."""

        def _text(parent, path):
            el = parent.find(path)
            return el.text.strip() if el is not None and el.text else None

        txn_date = _text(txn_el, ".//transactionDate/value")
        txn_code = _text(txn_el, ".//transactionCoding/transactionCode")
        shares_str = _text(txn_el, ".//transactionAmounts/transactionShares/value")
        price_str = _text(txn_el, ".//transactionAmounts/transactionPricePerShare/value")
        acq_disp = _text(txn_el, ".//transactionAmounts/transactionAcquiredDisposedCode/value")
        shares_after_str = _text(
            txn_el, ".//postTransactionAmounts/sharesOwnedFollowingTransaction/value"
        )
        ownership_type = _text(txn_el, ".//ownershipNature/directOrIndirectOwnership/value")

        if txn_date is None or shares_str is None:
            return None

        try:
            shares = float(shares_str)
        except (ValueError, TypeError):
            return None

        try:
            price = float(price_str) if price_str else None
        except (ValueError, TypeError):
            price = None

        try:
            shares_after = float(shares_after_str) if shares_after_str else None
        except (ValueError, TypeError):
            shares_after = None

        return {
            "ticker": ticker,
            "cik": cik,
            "insider_cik": insider_cik,
            "insider_name": insider_name,
            "insider_title": insider_title,
            "transaction_date": txn_date,
            "transaction_type": _TXN_CODES.get(txn_code, txn_code or "unknown"),
            "acquisition_disposition": acq_disp,
            "shares": shares,
            "price_per_share": price,
            "shares_after": shares_after,
            "ownership_type": ownership_type,
            "filing_date": filing_date,
        }

    def extract_to_raw(
        self,
        output_dir: Path,
        tickers: list[str] | None = None,
        ticker_to_cik: dict[str, int] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        run_id: str | None = None,
    ) -> list[Path]:
        """Extract insider trades for given tickers."""
        settings = get_settings()
        tickers = tickers or settings.prices.universe
        output_dir = Path(output_dir) / "sec_insider"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get CIK mapping if not provided
        if ticker_to_cik is None:
            from pipeline.extract.sec_fundamentals import SecFundamentalsExtractor

            ticker_to_cik = SecFundamentalsExtractor()._fetch_ticker_to_cik()

        saved_files: list[Path] = []

        for ticker in tickers:
            cik = ticker_to_cik.get(ticker.upper())
            if cik is None:
                logger.warning(f"No CIK for {ticker}, skipping insider trades")
                continue

            logger.info(f"Extracting insider trades for {ticker} (CIK {cik})")
            try:
                filings = self._fetch_filing_index(cik, start_date)
                all_rows: list[dict] = []

                for filing in filings[:100]:  # Cap at 100 filings per ticker
                    accession = filing.get("accession_number")
                    primary_doc = filing.get("primary_document", "")
                    f_date = filing.get("filing_date")

                    if not accession or not primary_doc.endswith(".xml"):
                        continue

                    xml_text = self._fetch_form4_xml(cik, accession, primary_doc)
                    if xml_text:
                        rows = self._parse_form4_xml(xml_text, ticker, cik, f_date or "")
                        for row in rows:
                            row["accession_number"] = accession
                        all_rows.extend(rows)

                    time.sleep(0.12)  # SEC rate limit

                if not all_rows:
                    continue

                df = pd.DataFrame(all_rows)
                df["extracted_at"] = datetime.now(UTC)
                df["run_id"] = run_id

                if start_date:
                    df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.date
                    df = df[df["transaction_date"] >= start_date]
                if end_date:
                    if "transaction_date" not in df.select_dtypes(include=["object"]).columns:
                        df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.date
                    df = df[df["transaction_date"] <= end_date]

                if df.empty:
                    continue

                file_path = output_dir / f"{ticker}_{start_date}_{end_date}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                self._metrics.record_extracted("sec_insider", len(df))
                logger.info(f"Saved {len(df)} insider trades for {ticker}")

            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed insider trades for {ticker}: {e}")
                continue

        return saved_files


def extract_sec_insider(
    output_dir: Path,
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """CLI-friendly wrapper."""
    extractor = SecInsiderExtractor()
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    return extractor.extract_to_raw(
        output_dir=output_dir,
        tickers=tickers,
        start_date=start,
        end_date=end,
        run_id=run_id,
    )
