"""Load CFPB complaint narratives for money-transfer, virtual-currency and payment-app products.

The live CFPB API and bulk export no longer include the consumer narrative field, so the
narratives come from a public-domain mirror of the last bulk export that still had them
(observation period 2011-12-01 to 2026-08-11). A sample of complaint ids is checked against
the live API to confirm that the mirror's metadata matches the official record.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import requests

PRODUCT = "Money transfer, virtual currency, or money service"
SNAPSHOT_REPO = "vivekkopthsd/cfpb-consumer-complaints"
SNAPSHOT_FILE = "complaints.parquet"
SNAPSHOT_SHA256 = "6e0a76bea4a9f5481aba4f895e5fa8f9a55df0751969971035637954c0a8067d"
LIVE_API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

COLUMNS = ["complaint_id", "date_received", "sub_product", "issue", "sub_issue", "company", "submitted_via", "narrative"]

_REDACTION = re.compile(r"\b[X]{2,}(?:/[X]{2,})*\b|\{\$|\}|year>")
_SPACES = re.compile(r"\s+")


def download_snapshot(raw_dir: Path) -> Path:
    """Download the snapshot parquet (~1 GB) into `raw_dir` once and verify its sha256."""
    from huggingface_hub import hf_hub_download

    path = raw_dir / SNAPSHOT_FILE
    marker = raw_dir / (SNAPSHOT_FILE + ".sha256-ok")
    if not path.exists():
        hf_hub_download(SNAPSHOT_REPO, SNAPSHOT_FILE, repo_type="dataset", local_dir=raw_dir)
    if not marker.exists():
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                digest.update(block)
        if digest.hexdigest() != SNAPSHOT_SHA256:
            raise ValueError(f"checksum mismatch for {path}")
        marker.write_text(SNAPSHOT_SHA256)
    return path


def clean_text(text: str) -> str:
    """Model input: drop CFPB redaction marks (XXXX, XX/XX/XXXX) and the braces around amounts."""
    text = _REDACTION.sub(" ", text.replace("{$", "$"))
    return _SPACES.sub(" ", text).strip()


def load_complaints(parquet_path: Path, start: str, end: str) -> pd.DataFrame:
    """Complaints of the money-transfer product family with a non-empty narrative, received in [start, end]."""
    table = pq.read_table(parquet_path, columns=COLUMNS, filters=[("product", "=", PRODUCT)])
    df = table.to_pandas()
    df = df[df["narrative"].fillna("").str.strip().str.len() > 0]
    df = df[(df["date_received"] >= pd.Timestamp(start)) & (df["date_received"] <= pd.Timestamp(end))]
    df = df.rename(columns={"date_received": "date"})
    df["complaint_id"] = df["complaint_id"].astype(str)
    df["month"] = df["date"].dt.to_period("M")
    df["text"] = df["narrative"].map(clean_text)
    for col in ("sub_product", "issue", "sub_issue", "company"):
        df[col] = df[col].fillna("(none)")
    return df.sort_values(["date", "complaint_id"]).reset_index(drop=True)


def fetch_live_record(complaint_id: str, timeout: float = 30.0, retries: int = 4) -> dict | None:
    """Metadata of one complaint from the live CFPB API (None if it is not found).

    The API rate-limits bursts of requests (HTTP 429); those are retried with a growing pause.
    """
    for attempt in range(retries + 1):
        r = requests.get(f"{LIVE_API}{complaint_id}", timeout=timeout)
        if r.status_code == 429 and attempt < retries:
            time.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
        return hits[0]["_source"] if hits else None
    return None


def live_crosscheck(df: pd.DataFrame, n: int, cache_path: Path, seed: int = 0) -> pd.DataFrame:
    """Compare product, issue, company and date of `n` sampled complaints with the live API.

    API responses are cached as JSON so re-runs make no network calls.
    """
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    rows = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for rec in sample.itertuples():
        if rec.complaint_id not in cache:
            cache[rec.complaint_id] = fetch_live_record(rec.complaint_id)
            cache_path.write_text(json.dumps(cache))  # keep progress if a later call fails
            time.sleep(1.5)
        live = cache[rec.complaint_id]
        rows.append(
            {
                "complaint_id": rec.complaint_id,
                "found_live": live is not None,
                "product_match": live is not None and live.get("product") == PRODUCT,
                "issue_match": live is not None and live.get("issue") == rec.issue,
                "company_match": live is not None and live.get("company") == rec.company,
                "date_match": live is not None and str(live.get("date_received", ""))[:10] == f"{rec.date:%Y-%m-%d}",
                "narrative_in_live_api": live is not None and bool(live.get("complaint_what_happened")),
            }
        )
    return pd.DataFrame(rows)
