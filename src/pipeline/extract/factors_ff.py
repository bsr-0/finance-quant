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
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        if not zf.namelist():
            raise ValueError("Empty ZIP file — no CSV found")
        name = zf.namelist()[0]
        with zf.open(name) as f:
            df = pd.read_csv(f, skiprows=3)
    df = df.rename(columns={"Unnamed: 0": "date"})
    df = df[df["date"].astype(str).str.len() == 8]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
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
