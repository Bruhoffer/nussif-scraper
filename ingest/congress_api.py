"""Congress trades ingest pipeline via RapidAPI.

Fetches the full historical trade data for all politicians available
from the politician-trade-tracker1.p.rapidapi.com API (Senate + House),
transforms the response into the shared Trade schema, and upserts into
the trades table.

Entry point: run_ingest()
"""

from __future__ import annotations

import os
import re
import time

import pandas as pd
import requests

from db.config import init_db
from db.prices import update_all_current_prices
from db.ticker_metadata import enrich_ticker_metadata
from ingest.common import clean_and_upsert

API_HOST = "politician-trade-tracker1.p.rapidapi.com"


def _api_headers() -> dict:
    api_key = os.getenv("RAPID_API_KEY")
    if not api_key:
        raise EnvironmentError("RAPID_API_KEY environment variable is not set.")
    return {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": API_HOST,
    }


def _get_all_politician_names() -> list[str]:
    """Fetch the directory of all politicians available in the API."""
    url = f"https://{API_HOST}/get_politicians"
    response = requests.get(url, headers=_api_headers())
    if response.status_code == 200:
        return list(response.json().keys())
    return []


def _get_full_profile(name: str) -> dict | None:
    """Fetch the complete historical trade data for a specific politician."""
    url = f"https://{API_HOST}/get_profile"
    response = requests.get(url, headers=_api_headers(), params={"name": name})
    if response.status_code == 200:
        return response.json()
    return None


def _parse_amount(amount_str: str | None) -> tuple[float | None, float | None, float | None]:
    """Parse a RapidAPI amount string like '1K-15K' or '>1M' into (min, max, midpoint)."""
    if pd.isna(amount_str) or not isinstance(amount_str, str):
        return None, None, None

    s = amount_str.strip()

    def _to_float(p: str) -> float:
        val = float(re.sub(r"[^\d.]", "", p))
        if "M" in p:
            val *= 1_000_000
        elif "K" in p:
            val *= 1_000
        return val

    if ">" in s:
        return _to_float(s), None, _to_float(s)

    parts = s.split("-")
    if len(parts) != 2:
        return None, None, None

    try:
        lo, hi = _to_float(parts[0]), _to_float(parts[1])
        return lo, hi, (lo + hi) / 2
    except (ValueError, IndexError):
        return None, None, None


def _clean_ticker(raw: str | None) -> str | None:
    """Normalise a raw ticker from the RapidAPI into a valid exchange symbol.

    The API occasionally concatenates company-type descriptors onto the
    front of tickers (e.g. 'INCKVUE' for Kenvue Inc, 'ETFIBIT' for the
    IBIT ETF). This function strips those known prefixes and fixes other
    common formatting issues.
    """
    if not isinstance(raw, str) or raw in ("N/A", "", "None"):
        return None

    # Strip exchange suffix: 'FCN:US' -> 'FCN'
    ticker = raw.split(":")[0].strip()

    # Known garbled prefixes to strip (longest first to avoid partial matches)
    _PREFIXES = (
        "CORPORATION",
        "CORP",
        "INC",
        "ETF",
        "PLC",
        "LTD",
    )
    for prefix in _PREFIXES:
        if ticker.startswith(prefix) and len(ticker) > len(prefix):
            ticker = ticker[len(prefix):]
            break  # only strip one prefix

    # Fix dot-notation to hyphen for preferred share classes (BRK.B -> BRK-B)
    ticker = ticker.replace(".", "-")

    return ticker if ticker else None


def _transform_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a RapidAPI trades DataFrame into the shared Trade schema."""
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()

    out["senator_name"] = df["politician_name"]
    out["senator_first_name"] = df["politician_name"].apply(
        lambda x: x.split()[0] if isinstance(x, str) else None
    )
    out["senator_last_name"] = df["politician_name"].apply(
        lambda x: " ".join(x.split()[1:]) if isinstance(x, str) and len(x.split()) > 1 else None
    )
    out["senator_display_name"] = df["politician_name"]
    out["chamber"] = df["chamber"]
    out["asset_name"] = df["company"]

    # Strip exchange suffix (e.g. 'FCN:US' -> 'FCN'), drop 'N/A',
    # then clean garbled prefixes the API concatenates from company type
    # descriptors (e.g. 'INCKVUE' -> 'KVUE', 'ETFIBIT' -> 'IBIT').
    out["ticker"] = df["ticker"].apply(_clean_ticker)

    out["transaction_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date

    # Estimate filing date from disclosure lag
    out["filing_date"] = out["transaction_date"] + pd.to_timedelta(
        pd.to_numeric(df["days_until_disclosure"], errors="coerce"), unit="D"
    )

    out["transaction_type"] = df["trade_type"]
    out["transaction_type_raw"] = df["trade_type"]
    out["amount_range_raw"] = df["trade_amount"]

    parsed = df["trade_amount"].apply(_parse_amount)
    out["amount_min"] = parsed.apply(lambda x: x[0])
    out["amount_max"] = parsed.apply(lambda x: x[1])
    out["mid_point"] = parsed.apply(lambda x: x[2])

    # Fields that don't exist in this data source
    for col in ("report_id", "report_type", "report_format", "owner", "asset_type", "comment"):
        out[col] = None

    return out


def run_ingest(limit_politicians: int | None = None) -> None:
    """Fetch all Congress trades from the RapidAPI and upsert to DB.

    Parameters
    ----------
    limit_politicians:
        Cap the number of politicians fetched. Useful for testing.
        Pass None (default) to fetch everyone.
    """

    init_db()

    print("[congress_api] Fetching politician directory...")
    names = _get_all_politician_names()

    if limit_politicians is not None:
        names = names[:limit_politicians]

    print(f"[congress_api] Found {len(names)} politicians. Starting full history pull...")

    all_trades: list[dict] = []
    for i, name in enumerate(names):
        print(f"[congress_api] [{i + 1}/{len(names)}] Pulling history for: {name}")
        profile = _get_full_profile(name)

        if profile and "Trade Data" in profile:
            for trade in profile["Trade Data"]:
                trade["politician_name"] = name
                all_trades.append(trade)

        time.sleep(1)  # respect rate limit

    if not all_trades:
        print("[congress_api] No trade data found.")
        return

    raw_df = pd.DataFrame(all_trades)
    raw_df["trade_date"] = pd.to_datetime(raw_df["trade_date"], errors="coerce")
    raw_df = raw_df.sort_values("trade_date", ascending=False)

    df = _transform_for_db(raw_df)

    inserted = clean_and_upsert(df)
    print(f"[congress_api] Upsert complete. Inserted {inserted} new trades (fetched {len(df)} total).")

    print("[congress_api] Updating current prices for all historical trades...")
    update_all_current_prices()

    print("[congress_api] Enriching ticker metadata (sectors & industries)...")
    enriched = enrich_ticker_metadata(max_tickers=None)
    print(f"[congress_api] Enriched metadata for {enriched} new tickers.")
