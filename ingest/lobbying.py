"""Ingest corporate lobbying disclosures from the Senate LDA API.

Source: https://lda.senate.gov/api/v1/filings/
No API key required. Filings are quarterly LD-2 disclosures.

Usage:
    python -m ingest.lobbying              # last 120 days
    python -m ingest.lobbying --days 365   # last year
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

import requests
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from db.config import SessionLocal, init_db
from db.models import LobbyingFiling
from ingest.ticker_mapper import map_company_to_ticker

_BASE_URL = "https://lda.senate.gov/api/v1/filings/"
_PAGE_SIZE = 25
_REQUEST_DELAY = 2.0      # seconds between pages
_MAX_RETRIES = 5
_BACKOFF_BASE = 10        # seconds for first retry on 429


def _fetch_url(url: str, params: dict | None = None) -> dict:
    """GET ``url`` with retry/backoff on 429."""
    for attempt in range(1, _MAX_RETRIES + 1):
        resp = requests.get(url, params=params, timeout=30)

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() \
                else _BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"[lobbying] Rate limited (429). Waiting {wait}s before retry "
                  f"{attempt}/{_MAX_RETRIES}...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp.json()

    raise requests.RequestException(f"Exceeded {_MAX_RETRIES} retries due to rate limiting.")


def _build_issues_summary(activities: list[dict]) -> str:
    """Flatten lobbying_activities into a short readable string."""
    parts = []
    for act in activities or []:
        code = act.get("general_issue_code_display", "")
        desc = (act.get("description") or "").strip()
        if code and desc:
            parts.append(f"{code}: {desc[:120]}")
        elif code:
            parts.append(code)
    summary = " | ".join(parts)
    return summary[:2000]   # match column size


def _period_label(filing_type: str | None, filing_year: int | None) -> str:
    """Convert LDA filing_type code to a human-readable period label."""
    mapping = {
        "Q1": "Q1", "Q2": "Q2", "Q3": "Q3", "Q4": "Q4",
        "MM": "Mid-Year", "YE": "Year-End",
        "RR": "Registration",
    }
    label = mapping.get(filing_type or "", filing_type or "")
    if filing_year:
        return f"{label} {filing_year}"
    return label


def _upsert_filing(session, data: dict) -> bool:
    """Insert or update one LobbyingFiling row. Returns True if new."""
    uuid = data.get("filing_uuid")
    if not uuid:
        return False

    existing = session.execute(
        select(LobbyingFiling).where(LobbyingFiling.filing_uuid == uuid)
    ).scalars().first()

    client_name = (data.get("client") or {}).get("name") or ""
    registrant_name = (data.get("registrant") or {}).get("name") or ""
    amount_raw = data.get("income") or data.get("expenses")
    try:
        amount = float(amount_raw) if amount_raw else None
    except (TypeError, ValueError):
        amount = None

    filing_date_str = data.get("dt_posted") or data.get("filing_date") or ""
    try:
        filing_date = dt.date.fromisoformat(filing_date_str[:10]) if filing_date_str else None
    except ValueError:
        filing_date = None

    ticker = map_company_to_ticker(client_name) if client_name else None
    issues = _build_issues_summary(data.get("lobbying_activities") or [])
    period = _period_label(data.get("filing_type"), data.get("filing_year"))
    source_url = f"https://lda.senate.gov/filings/public/filing/{uuid}/print/"

    if existing is None:
        session.add(LobbyingFiling(
            filing_uuid=uuid,
            client_name=client_name,
            ticker=ticker,
            registrant_name=registrant_name,
            amount=amount,
            filing_date=filing_date,
            period_of_lobbying=period,
            specific_issues=issues,
            source_url=source_url,
        ))
        return True
    else:
        existing.client_name = client_name
        existing.ticker = ticker
        existing.registrant_name = registrant_name
        existing.amount = amount
        existing.filing_date = filing_date
        existing.period_of_lobbying = period
        existing.specific_issues = issues
        return False


def run_ingest(days: int = 120) -> None:
    init_db()
    after_date = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    print(f"[lobbying] Fetching filings posted after {after_date}...")

    # First page uses query params; subsequent pages follow the cursor `next` URL
    # returned by the API — this avoids pagination drift on live data.
    next_url: str | None = _BASE_URL
    first_page_params: dict | None = {
        "filing_dt_posted_after": after_date,
        "limit": _PAGE_SIZE,
        "ordering": "-filing_dt_posted",   # stable newest-first ordering
    }

    total_fetched = 0
    new_rows = 0

    while next_url:
        try:
            page = _fetch_url(next_url, params=first_page_params)
            first_page_params = None   # only send params on the first request
        except requests.RequestException as exc:
            print(f"[lobbying] HTTP error: {exc}")
            break

        results = page.get("results") or []
        if not results:
            break

        try:
            with SessionLocal() as session:
                for item in results:
                    if _upsert_filing(session, item):
                        new_rows += 1
                session.commit()
        except SQLAlchemyError as exc:
            print(f"[lobbying] DB error: {exc}")
            break

        total_fetched += len(results)
        count = page.get("count") or 0
        print(f"[lobbying] {total_fetched}/{count} filings processed ({new_rows} new)...")

        next_url = page.get("next") or None
        if next_url:
            time.sleep(_REQUEST_DELAY)

    print(f"[lobbying] Done. {total_fetched} fetched, {new_rows} new rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest LDA lobbying filings")
    parser.add_argument("--days", type=int, default=120,
                        help="How many days back to fetch (default: 120)")
    args = parser.parse_args()
    run_ingest(days=args.days)


if __name__ == "__main__":
    main()
