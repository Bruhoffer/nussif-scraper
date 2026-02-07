"""Scrape PTR trades over a date range and upsert into the local DB.

Usage (from project root, with env activated):

    cd "insiderscraper"
    python ingest_ptr_trades.py --days 90
"""

from __future__ import annotations

import argparse
import datetime as dt

from dotenv import load_dotenv

# Load environment variables from .env so DATABASE_URL (and others)
# are available before we import db.config, which constructs the
# SQLAlchemy engine.
load_dotenv()

from scraper.pipeline import fetch_ptr_trades_for_range
from db.config import init_db
from db.upsert import upsert_trades


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape PTR trades and upsert into DB")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="How many days back from today to fetch filings",
    )
    args = parser.parse_args()

    # Ensure tables exist
    init_db()

    today = dt.date.today()
    start_date = today - dt.timedelta(days=args.days)

    print(f"Fetching PTR trades filed between {start_date} and {today}...")
    df = fetch_ptr_trades_for_range(start_date, today)

    if df.empty:
        print("No PTR trades found in this range. Nothing to upsert.")
        return

    trades = df.to_dict(orient="records")
    inserted = upsert_trades(trades)
    print(f"Upsert complete. Inserted {inserted} new trades (scraped {len(trades)} total).")


if __name__ == "__main__":
    main()
