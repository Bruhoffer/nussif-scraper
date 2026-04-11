"""Ingest SEC Schedule 13D/G activist and passive beneficial-ownership filings.

Source: sec-api.io Form 13D/13G API (https://api.sec-api.io/form-13d-13g)
Requires: SEC_API_KEY in .env

Provides fully structured JSON — no SGML/HTML parsing needed.
New filings are added within 500ms of publication on EDGAR.

Usage:
    python -m ingest.activist_filings             # last 90 days
    python -m ingest.activist_filings --days 365
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time

import requests
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from db.config import SessionLocal, init_db
from db.models import ActivistFiling
from ingest.ticker_mapper import map_company_to_ticker

_API_URL = "https://api.sec-api.io/form-13d-13g"
_PAGE_SIZE = 50          # sec-api.io max is 50
_MAX_RECORDS = 2000
_REQUEST_DELAY = 0.3     # seconds between pages


def _api_key() -> str:
    key = os.getenv("SEC_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "SEC_API_KEY not set. Add it to your .env file.\n"
            "Get a free key at https://sec-api.io"
        )
    return key


def _fetch_page(start_date: str, end_date: str, offset: int) -> dict:
    payload = {
        "query": f'filedAt:[{start_date} TO {end_date}]',
        "from": str(offset),
        "size": str(_PAGE_SIZE),
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    resp = requests.post(
        _API_URL,
        json=payload,
        headers={"Authorization": _api_key()},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_investor(filing: dict) -> str:
    """Return the lead investor name (filer, not subject company)."""
    # Prefer owners[0].name — the primary reporting person
    owners = filing.get("owners") or []
    if owners:
        name = owners[0].get("name")
        if name:
            # API sometimes returns name as a list
            if isinstance(name, list):
                name = name[0] if name else ""
            return name

    # Fallback: filer tagged "(Filed by)"
    for filer in (filing.get("filers") or []):
        name = filer.get("name") or ""
        if "(Filed by)" in name:
            return name.replace("(Filed by)", "").strip()

    # Last resort: first filer
    filers = filing.get("filers") or []
    return filers[0].get("name", "") if filers else ""


def _upsert_filing(session, filing: dict) -> bool:
    """Insert or update one ActivistFiling row. Returns True if new."""
    accession_no = filing.get("accessionNo") or ""
    if not accession_no:
        return False

    existing = session.execute(
        select(ActivistFiling).where(ActivistFiling.accession_number == accession_no)
    ).scalars().first()

    target_company = filing.get("nameOfIssuer") or ""
    ticker = map_company_to_ticker(target_company) if target_company else None
    lead_investor = _extract_investor(filing)
    form_type = filing.get("formType") or ""

    filed_at = filing.get("filedAt") or ""
    try:
        filing_date = dt.date.fromisoformat(filed_at[:10]) if filed_at else None
    except ValueError:
        filing_date = None

    # Primary owner data (aggregate across all reporting persons)
    owners = filing.get("owners") or []
    primary = owners[0] if owners else {}
    shares = primary.get("aggregateAmountOwned")
    ownership_pct = primary.get("amountAsPercent")

    try:
        shares = float(shares) if shares is not None else None
        ownership_pct = float(ownership_pct) if ownership_pct is not None else None
    except (TypeError, ValueError):
        shares = None
        ownership_pct = None

    sec_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{(filing.get('filers') or [{}])[0].get('cik', '')}/"
        f"{accession_no.replace('-', '')}/{accession_no}-index.htm"
    )

    if existing is None:
        session.add(ActivistFiling(
            accession_number=accession_no,
            ticker=ticker,
            target_company=target_company,
            lead_investor=lead_investor,
            form_type=form_type,
            filing_date=filing_date,
            shares=shares,
            prev_shares=None,
            ownership_pct=ownership_pct,
            shares_change_pct=None,
            market_cap_m=None,
            sec_url=sec_url,
        ))
        return True
    else:
        existing.ticker = ticker
        existing.target_company = target_company
        existing.lead_investor = lead_investor
        existing.form_type = form_type
        existing.filing_date = filing_date
        existing.shares = shares
        existing.ownership_pct = ownership_pct
        existing.sec_url = sec_url
        return False


def run_ingest(days: int = 90) -> None:
    init_db()
    end_date = dt.date.today().isoformat()
    start_date = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    print(f"[activist_filings] Fetching 13D/G filings from {start_date} to {end_date}...")

    offset = 0
    total_fetched = 0
    new_rows = 0

    while total_fetched < _MAX_RECORDS:
        try:
            data = _fetch_page(start_date, end_date, offset)
        except requests.RequestException as exc:
            print(f"[activist_filings] API error at offset {offset}: {exc}")
            break

        filings = data.get("filings") or []
        if not filings:
            break

        for filing in filings:
            try:
                with SessionLocal() as session:
                    if _upsert_filing(session, filing):
                        new_rows += 1
                    session.commit()
            except SQLAlchemyError as exc:
                print(f"[activist_filings] DB error for {filing.get('accessionNo')}: {exc}")

        total_fetched += len(filings)
        total_available = (data.get("total") or {}).get("value", "?")
        print(f"[activist_filings] {total_fetched}/{total_available} filings processed "
              f"({new_rows} new)...")

        if len(filings) < _PAGE_SIZE:
            break

        offset += _PAGE_SIZE
        time.sleep(_REQUEST_DELAY)

    print(f"[activist_filings] Done. {total_fetched} fetched, {new_rows} new rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SEC 13D/G activist filings")
    parser.add_argument("--days", type=int, default=90,
                        help="How many days back to fetch (default: 90)")
    args = parser.parse_args()
    run_ingest(days=args.days)


if __name__ == "__main__":
    main()
