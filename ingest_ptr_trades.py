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

try:
    # Preferred imports when insiderscraper is installed as a package
    # (e.g. in Azure Functions, where `insiderscraper` is a top-level
    # package and `scraper` / `db` live under it).
    from .scraper.pipeline import fetch_ptr_trades_for_range
    from .db.config import init_db
    from .db.upsert import upsert_trades
    from .db.prices import enrich_prices_for_trades
except ImportError:  # Fallback for running as a script: `python ingest_ptr_trades.py`
    from scraper.pipeline import fetch_ptr_trades_for_range
    from db.config import init_db
    from db.upsert import upsert_trades
    from db.prices import enrich_prices_for_trades


def run_ingest(days: int = 90) -> None:
    """Run a single PTR ingest for the last ``days`` days of filings.

    This is the core pipeline used by both the CLI entrypoint and any
    scheduled jobs (e.g. Azure Functions Timer Trigger).
    """

    # Ensure tables exist
    init_db()

    today = dt.date.today()
    start_date = today - dt.timedelta(days=days)

    print(f"Fetching PTR trades filed between {start_date} and {today}...")
    df = fetch_ptr_trades_for_range(start_date, today)

    if df.empty:
        print("No PTR trades found in this range. Nothing to upsert.")
        return

    # Enrich with historical and latest prices before persisting, so
    # downstream analytics and the Streamlit app can rely purely on the
    # DB without calling yfinance.
    enrich_prices_for_trades(df)

    trades = df.to_dict(orient="records")
    inserted = upsert_trades(trades)
    print(f"Upsert complete. Inserted {inserted} new trades (scraped {len(trades)} total).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape PTR trades and upsert into DB")
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="How many days back from today to fetch filings",
    )
    args = parser.parse_args()

    run_ingest(days=args.days)


if __name__ == "__main__":
    main()
