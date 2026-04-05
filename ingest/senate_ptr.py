"""Senate PTR ingest pipeline.

Scrapes the Senate eFD website for Periodic Transaction Reports filed
in the last N days, enriches prices, and upserts into the trades table.

Entry point: run_ingest(days)
"""

from __future__ import annotations

import datetime as dt

from db.config import init_db
from db.prices import update_all_current_prices
from db.ticker_metadata import enrich_ticker_metadata
from ingest.common import clean_and_upsert
from scraper.pipeline import fetch_ptr_trades_for_range


def run_ingest(days: int = 3) -> None:
    """Scrape Senate PTR trades for the last ``days`` days and upsert to DB.

    Steps:
    1. Ensure DB tables exist.
    2. Fetch all PTR reports filed in [today - days, today] from the Senate eFD site.
    3. Enrich each trade with price_at_transaction and current_price.
    4. Clean and upsert into the trades table (idempotent).
    5. Refresh current_price for all historical trades.
    6. Enrich any new tickers with sector / industry metadata.
    """

    init_db()

    today = dt.date.today()
    start_date = today - dt.timedelta(days=days)

    print(f"[senate_ptr] Fetching PTR trades filed between {start_date} and {today}...")
    df = fetch_ptr_trades_for_range(start_date, today)

    if df.empty:
        print("[senate_ptr] No PTR trades found in this range.")
        return

    inserted = clean_and_upsert(df)
    print(f"[senate_ptr] Upsert complete. Inserted {inserted} new trades (scraped {len(df)} total).")

    print("[senate_ptr] Updating current prices for all historical trades...")
    update_all_current_prices()

    print("[senate_ptr] Enriching ticker metadata (sectors & industries)...")
    enriched = enrich_ticker_metadata(max_tickers=None)
    print(f"[senate_ptr] Enriched metadata for {enriched} new tickers.")
