"""Fama-French factor data extractor (FF5 + Momentum)."""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

_MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)


def _read_zip_csv(content: bytes) -> pd.DataFrame:
    """Parse a Ken French factor ZIP → DataFrame indexed by date.

    Ken French files have a variable-length text preamble and frequently
    concatenate multiple tables (daily + annual, etc.) in a single CSV,
    separated by blank lines and text section headers. We locate the
    header row dynamically and stop at the first blank line or non-date
    row after the data starts.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        if not zf.namelist():
            raise ValueError("Empty ZIP file — no CSV found")
        name = zf.namelist()[0]
        with zf.open(name) as f:
            raw = f.read().decode("utf-8", errors="replace")

    lines = raw.splitlines()

    # Header row: first line that starts with ',' and contains column letters
    header_idx: int | None = None
    for i, line in enumerate(lines):
        if line.startswith(",") and any(c.isalpha() for c in line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not locate Fama-French CSV header row")

    # Data block ends at first blank line or first row whose leading field
    # isn't an 8-digit YYYYMMDD date (handles daily→annual table boundary).
    end_idx = len(lines)
    for i in range(header_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            end_idx = i
            break
        first_field = stripped.split(",")[0].strip()
        if not (first_field.isdigit() and len(first_field) == 8):
            end_idx = i
            break

    csv_block = "\n".join(lines[header_idx:end_idx])
    df = pd.read_csv(io.StringIO(csv_block))
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date")
    return df


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _download(url: str) -> bytes:
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    return response.content


def extract_factors_ff(output_dir: Path, run_id: str | None = None) -> list[Path]:
    """Extract FF5 + Momentum factors and save to raw parquet."""
    output_dir = Path(output_dir) / "factors"
    output_dir.mkdir(parents=True, exist_ok=True)

    ff5_zip = _download(_FF5_URL)
    mom_zip = _download(_MOM_URL)

    ff5 = _read_zip_csv(ff5_zip)
    mom = _read_zip_csv(mom_zip)

    df = ff5.join(mom, how="inner")
    df = df.rename(
        columns={
            "Mkt-RF": "mkt_rf",
            "SMB": "smb",
            "HML": "hml",
            "RMW": "rmw",
            "CMA": "cma",
            "Mom   ": "mom",
            "Mom": "mom",
            "RF": "rf",
        }
    )

    for col in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df = df[["mkt_rf", "smb", "hml", "rmw", "cma", "mom", "rf"]].dropna()
    df = df.reset_index()
    df["extracted_at"] = datetime.now(UTC)
    df["run_id"] = run_id

    file_path = output_dir / "ff_factors_daily.parquet"
    df.to_parquet(file_path, index=False)
    logger.info(f"Saved {len(df)} factor rows to {file_path}")
    return [file_path]
