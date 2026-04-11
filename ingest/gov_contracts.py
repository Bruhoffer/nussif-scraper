"""Ingest federal contract awards from USASpending.gov.

Source: https://api.usaspending.gov (free, no API key required)
Fetches the top 500 contract awards by amount for the requested date window.

Usage:
    python -m ingest.gov_contracts             # last 90 days
    python -m ingest.gov_contracts --days 365
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import requests
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from db.config import SessionLocal, init_db
from db.models import GovContract
from ingest.ticker_mapper import map_company_to_ticker

_SEARCH_URL = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/"
_PAGE_SIZE = 100
_MAX_RECORDS = 1000     # cap to avoid enormous payloads
_REQUEST_DELAY = 0.5    # seconds between pages


def _fetch_page(start_date: str, end_date: str, page: int) -> dict:
    # spending_by_transaction gives one row per contract action with a real
    # Action Date — the spending_by_award endpoint returns null dates for
    # long-running multi-year contracts.
    payload = {
        "filters": {
            "award_type_codes": ["A", "B", "C", "D"],   # all contract types
            "time_period": [{"start_date": start_date, "end_date": end_date}],
        },
        "fields": [
            "Award ID",
            "Recipient Name",
            "Action Date",
            "Transaction Amount",
            "Awarding Agency",
            "Transaction Description",
        ],
        "page": page,
        "limit": _PAGE_SIZE,
        "sort": "Transaction Amount",
        "order": "desc",
    }
    resp = requests.post(_SEARCH_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _upsert_contract(session, row: dict) -> bool:
    """Insert or update one GovContract row. Returns True if new."""
    # internal_id is unique per transaction row; generated_internal_id is
    # per award and repeats across modifications — can't use it as PK.
    award_id = str(row.get("internal_id") or "")
    if not award_id or award_id == "0":
        return False

    existing = session.execute(
        select(GovContract).where(GovContract.award_id == award_id)
    ).scalars().first()

    recipient_name = row.get("Recipient Name") or ""
    ticker = map_company_to_ticker(recipient_name) if recipient_name else None

    award_amount = row.get("Transaction Amount") or row.get("Award Amount")
    try:
        award_amount = float(award_amount) if award_amount is not None else None
    except (TypeError, ValueError):
        award_amount = None

    award_date_str = row.get("Action Date") or row.get("Award Date") or ""
    try:
        award_date = dt.date.fromisoformat(award_date_str[:10]) if award_date_str else None
    except ValueError:
        award_date = None

    agency_raw = row.get("Awarding Agency") or ""
    funding_agency = (
        agency_raw.get("name", "") if isinstance(agency_raw, dict) else str(agency_raw)
    )

    description = (row.get("Transaction Description") or row.get("Description") or "")[:1000]

    if existing is None:
        session.add(GovContract(
            award_id=award_id,
            recipient_name=recipient_name,
            ticker=ticker,
            award_amount=award_amount,
            award_date=award_date,
            funding_agency=funding_agency,
            description=description,
        ))
        return True
    else:
        existing.recipient_name = recipient_name
        existing.ticker = ticker
        existing.award_amount = award_amount
        existing.award_date = award_date
        existing.funding_agency = funding_agency
        existing.description = description
        return False


def run_ingest(days: int = 90) -> None:
    init_db()
    end_date = dt.date.today().isoformat()
    start_date = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    print(f"[gov_contracts] Fetching contracts from {start_date} to {end_date}...")

    page = 1
    total_fetched = 0
    new_rows = 0

    while total_fetched < _MAX_RECORDS:
        try:
            data = _fetch_page(start_date, end_date, page)
        except requests.RequestException as exc:
            print(f"[gov_contracts] HTTP error on page {page}: {exc}")
            break

        results = (data.get("results") or [])
        if not results:
            break

        for row in results:
            try:
                with SessionLocal() as session:
                    if _upsert_contract(session, row):
                        new_rows += 1
                    session.commit()
            except SQLAlchemyError as exc:
                # Skip individual bad rows (e.g. rare constraint edge cases)
                pass

        total_fetched += len(results)
        meta = data.get("page_metadata") or {}
        total_available = meta.get("count", meta.get("total", "?"))
        print(f"[gov_contracts] {total_fetched}/{total_available} contracts processed ({new_rows} new)...")

        has_next = meta.get("hasNext", False)
        if not has_next or len(results) < _PAGE_SIZE:
            break

        page += 1
        time.sleep(_REQUEST_DELAY)

    print(f"[gov_contracts] Done. {total_fetched} fetched, {new_rows} new rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest USASpending.gov contract awards")
    parser.add_argument("--days", type=int, default=90,
                        help="How many days back to fetch (default: 90)")
    args = parser.parse_args()
    run_ingest(days=args.days)


if __name__ == "__main__":
    main()
