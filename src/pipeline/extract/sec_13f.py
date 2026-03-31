"""SEC EDGAR 13F institutional holdings extractor."""

from __future__ import annotations

import contextlib
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

logger = logging.getLogger(__name__)

SEC_USER_AGENT = "MarketDataWarehouse/1.0 (research; contact@example.com)"

# Well-known institutional filers to track (CIKs).
DEFAULT_FILERS: list[dict] = [
    {"name": "Berkshire Hathaway", "cik": 1067983},
    {"name": "Bridgewater Associates", "cik": 1350694},
    {"name": "Renaissance Technologies", "cik": 1037389},
    {"name": "Citadel Advisors", "cik": 1423053},
    {"name": "Two Sigma Investments", "cik": 1179392},
    {"name": "DE Shaw", "cik": 1009207},
    {"name": "AQR Capital Management", "cik": 1167557},
    {"name": "BlackRock", "cik": 1364742},
    {"name": "Vanguard Group", "cik": 102909},
    {"name": "State Street", "cik": 93751},
]

# 13F XML namespace
_NS = {"ns": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}


class Sec13FExtractor(HttpClientMixin):
    """Extract institutional holdings from SEC 13F filings."""

    def __init__(self) -> None:
        self.client = httpx.Client(
            timeout=60.0,
            headers={"User-Agent": SEC_USER_AGENT},
        )
        self._circuit = get_circuit_breaker(
            "sec_edgar_13f", failure_threshold=5, recovery_timeout=60.0
        )
        self._metrics = PipelineMetrics("sec_13f_extractor")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_submissions(self, cik: int) -> dict:
        """Fetch filing submissions index."""

        def _do() -> dict:
            padded = str(cik).zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{padded}.json"
            resp = self.client.get(url)
            resp.raise_for_status()
            return resp.json()

        return self._circuit.call(_do)

    def _get_13f_filings(self, cik: int, start_date: date | None = None) -> list[dict]:
        """Get list of 13F-HR filings for a filer."""
        data = self._fetch_submissions(cik)
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        filings = []
        for i, form in enumerate(forms):
            if form not in ("13F-HR", "13F-HR/A"):
                continue
            filing_date = dates[i] if i < len(dates) else None
            if start_date and filing_date and filing_date < start_date.isoformat():
                continue
            filings.append(
                {
                    "accession_number": accessions[i] if i < len(accessions) else None,
                    "filing_date": filing_date,
                    "primary_document": primary_docs[i] if i < len(primary_docs) else None,
                    "report_date": report_dates[i] if i < len(report_dates) else None,
                }
            )

        return filings

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    def _fetch_13f_table(self, cik: int, accession: str) -> str | None:
        """Fetch the information table XML for a 13F filing."""

        def _do() -> str | None:
            padded = str(cik).zfill(10)
            acc_clean = accession.replace("-", "")
            # Try to find the infotable document
            index_url = f"https://www.sec.gov/Archives/edgar/data/{padded}/{acc_clean}/"
            resp = self.client.get(index_url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            # Look for infotable.xml in the index
            text = resp.text.lower()
            # Common naming patterns for the info table
            for pattern in ["infotable.xml", "information_table.xml", "13f_infotable.xml"]:
                if pattern in text:
                    table_url = f"{index_url}{pattern}"
                    table_resp = self.client.get(table_url)
                    if table_resp.status_code == 200:
                        return table_resp.text
            return None

        return self._circuit.call(_do)

    @staticmethod
    def _parse_13f_xml(
        xml_text: str,
        filer_cik: int,
        filer_name: str,
        report_date: str,
        filing_date: str,
    ) -> list[dict]:
        """Parse 13F information table XML into holding records."""
        rows: list[dict] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return rows

        # Try with and without namespace
        info_tables = root.findall(".//ns:infoTable", _NS)
        if not info_tables:
            info_tables = root.findall(
                ".//{http://www.sec.gov/edgar/document/thirteenf/informationtable}infoTable"
            )
        if not info_tables:
            # Try without namespace
            info_tables = root.findall(".//infoTable")

        for entry in info_tables:

            def _text(path: str, entry: ET.Element = entry) -> str | None:
                # Try with namespace
                el = entry.find(f"ns:{path}", _NS)
                if el is None:
                    el = entry.find(
                        f"{{http://www.sec.gov/edgar/document/thirteenf/informationtable}}{path}"
                    )
                if el is None:
                    el = entry.find(path)
                return el.text.strip() if el is not None and el.text else None

            issuer = _text("nameOfIssuer")
            cusip = _text("cusip")
            class_title = _text("titleOfClass")
            value_str = _text("value")
            put_call = _text("putCall")
            discretion = _text("investmentDiscretion")

            # Shares info
            shares_el = entry.find("ns:shrsOrPrnAmt", _NS)
            if shares_el is None:
                shares_el = entry.find(
                    "{http://www.sec.gov/edgar/document/thirteenf/informationtable}shrsOrPrnAmt"
                )
            if shares_el is None:
                shares_el = entry.find("shrsOrPrnAmt")

            shares = None
            shares_type = None
            if shares_el is not None:
                sh_val = shares_el.find("ns:sshPrnamt", _NS)
                if sh_val is None:
                    sh_val = shares_el.find(
                        "{http://www.sec.gov/edgar/document/thirteenf/informationtable}sshPrnamt"
                    )
                if sh_val is None:
                    sh_val = shares_el.find("sshPrnamt")
                sh_type = shares_el.find("ns:sshPrnamtType", _NS)
                if sh_type is None:
                    sh_type = shares_el.find(
                        "{http://www.sec.gov/edgar/document/thirteenf/informationtable}sshPrnamtType"
                    )
                if sh_type is None:
                    sh_type = shares_el.find("sshPrnamtType")

                if sh_val is not None and sh_val.text:
                    with contextlib.suppress(ValueError):
                        shares = int(sh_val.text.strip())
                if sh_type is not None and sh_type.text:
                    shares_type = sh_type.text.strip()

            # Voting authority
            voting_el = entry.find("ns:votingAuthority", _NS)
            if voting_el is None:
                voting_el = entry.find(
                    "{http://www.sec.gov/edgar/document/thirteenf/informationtable}votingAuthority"
                )
            if voting_el is None:
                voting_el = entry.find("votingAuthority")

            vote_sole = vote_shared = vote_none = None
            if voting_el is not None:
                for tag, attr in [("Sole", "sole"), ("Shared", "shared"), ("None", "none")]:
                    vel = voting_el.find(f"ns:{attr}", _NS)
                    if vel is None:
                        vel = voting_el.find(
                            f"{{http://www.sec.gov/edgar/document/thirteenf/informationtable}}{attr}"
                        )
                    if vel is None:
                        vel = voting_el.find(attr)
                    if vel is not None and vel.text:
                        try:
                            val = int(vel.text.strip())
                        except ValueError:
                            val = None
                        if tag == "Sole":
                            vote_sole = val
                        elif tag == "Shared":
                            vote_shared = val
                        else:
                            vote_none = val

            market_value = None
            if value_str:
                with contextlib.suppress(ValueError):
                    market_value = int(value_str) * 1000  # 13F reports values in thousands

            rows.append(
                {
                    "filer_cik": filer_cik,
                    "filer_name": filer_name,
                    "report_date": report_date,
                    "filing_date": filing_date,
                    "cusip": cusip,
                    "issuer_name": issuer,
                    "class_title": class_title,
                    "market_value": market_value,
                    "shares_held": shares,
                    "shares_type": shares_type,
                    "put_call": put_call,
                    "investment_discretion": discretion,
                    "voting_authority_sole": vote_sole,
                    "voting_authority_shared": vote_shared,
                    "voting_authority_none": vote_none,
                }
            )

        return rows

    def extract_to_raw(
        self,
        output_dir: Path,
        filers: list[dict] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        run_id: str | None = None,
    ) -> list[Path]:
        """Extract 13F holdings for a list of institutional filers."""
        filers = filers or DEFAULT_FILERS
        output_dir = Path(output_dir) / "sec_13f"
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files: list[Path] = []

        for filer in filers:
            filer_name = filer["name"]
            filer_cik = filer["cik"]
            logger.info(f"Extracting 13F holdings for {filer_name} (CIK {filer_cik})")

            try:
                filings = self._get_13f_filings(filer_cik, start_date)
                all_rows: list[dict] = []

                for filing in filings[:8]:  # Cap at ~2 years of quarterly filings
                    accession = filing.get("accession_number")
                    report_date = filing.get("report_date")
                    filing_date = filing.get("filing_date")

                    if not accession:
                        continue

                    xml_text = self._fetch_13f_table(filer_cik, accession)
                    if xml_text:
                        rows = self._parse_13f_xml(
                            xml_text,
                            filer_cik,
                            filer_name,
                            str(report_date or ""),
                            str(filing_date or ""),
                        )
                        for row in rows:
                            row["accession_number"] = accession
                        all_rows.extend(rows)

                    time.sleep(0.12)

                if not all_rows:
                    continue

                df = pd.DataFrame(all_rows)
                df["extracted_at"] = datetime.now(UTC)
                df["run_id"] = run_id

                safe_name = filer_name.replace(" ", "_").lower()
                file_path = output_dir / f"{safe_name}_{start_date}_{end_date}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files.append(file_path)
                self._metrics.record_extracted("sec_13f", len(df))
                logger.info(f"Saved {len(df)} holdings for {filer_name}")

            except Exception as e:
                self._metrics.record_error(type(e).__name__)
                logger.error(f"Failed 13F for {filer_name}: {e}")
                continue

        return saved_files


def extract_sec_13f(
    output_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    run_id: str | None = None,
) -> list[Path]:
    """CLI-friendly wrapper."""
    extractor = Sec13FExtractor()
    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    return extractor.extract_to_raw(
        output_dir=output_dir,
        start_date=start,
        end_date=end,
        run_id=run_id,
    )
