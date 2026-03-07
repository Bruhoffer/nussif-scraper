"""Backfill script to ingest ~5 years of Senate PTR trades into the local DB.

Usage (from project root, with env activated):

    cd "insiderscraper"
    python backfill_ptr_5y.py --years 5

This is a thin wrapper around ``ingest_ptr_trades.run_ingest`` that simply
requests a long lookback window. It reuses the existing pipeline, including:

* Senate eFD scraping (report search + PTR detail pages)
* price enrichment via yfinance (price_at_transaction, current_price)
* idempotent upserts into the ``trades`` table

You can safely rerun this script; existing trades will be skipped thanks to
the UNIQUE constraint and upsert logic.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

# Ensure DATABASE_URL (and any other secrets) from .env are loaded before
# we import the ingest module, which constructs the SQLAlchemy engine.
load_dotenv()

try:
    # Package-style import (when insiderscraper is installed as a package)
    from .ingest_ptr_trades import run_ingest
except ImportError:  # Script-style fallback: ``python backfill_ptr_5y.py``
    from ingest_ptr_trades import run_ingest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill Senate PTR trades for the last N years",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="How many years back from today to fetch filings (approximate)",
    )
    args = parser.parse_args()

    # Approximate conversion; good enough for a backfill and keeps the code
    # simple. If you ever need exact date control, we can switch to explicit
    # start_date / end_date windowing.
    days = args.years * 365

    print(f"[backfill_ptr_5y] Starting backfill for last {args.years} years (~{days} days)...")
    run_ingest(days=days)
    print("[backfill_ptr_5y] Backfill complete.")


if __name__ == "__main__":
    main()
