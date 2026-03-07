"""Scrape PTR trades over a date range and upsert into the local DB.

Usage (from project root, with env activated):

    cd "insiderscraper"
    python ingest_ptr_trades.py --days 90
"""

from __future__ import annotations

import argparse
import datetime as dt
import numpy as np
import pandas as pd

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
    from db.prices import enrich_prices_for_trades, update_all_current_prices
    from db.ticker_metadata import enrich_ticker_metadata


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
    # DB without calling yfinance. ``enrich_prices_for_trades`` is
    # best-effort and only leaves NULL prices for tickers /
    # (ticker, date) pairs where yfinance failed, keeping other
    # trades fully usable.
    failed_tickers, failed_pairs = enrich_prices_for_trades(df)

    if failed_tickers or failed_pairs:
        print(
            "[ingest_ptr_trades] Pricing missing for "
            f"{len(failed_tickers)} tickers and {len(failed_pairs)} "
            "(ticker, transaction_date) pairs. "
            "These trades will be stored with NULL price fields."
        )

    # --- CLEANING STEP BEFORE UPSERT ---------------------------------
    # Replace NaN with None (SQL NULL)
    df = df.replace({np.nan: None})

    # Round price columns to 4 decimal places to prevent scale errors
    price_cols = ["price_at_transaction", "current_price"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round(x, 4) if x is not None else None)

    # 1. Force convert everything to standard Python types (removes
    # numpy-specific dtypes that SQLAlchemy/DB drivers may not like).
    df = df.astype(object)

    # 2. Replace all forms of "Not a Number" with None (SQL NULL)
    df = df.replace({np.nan: None, float("inf"): None, float("-inf"): None})

    # 3. Ensure no hidden "NaN" strings or objects remain
    df = df.where(pd.notnull(df), None)

    trades = df.to_dict(orient="records")
    inserted = upsert_trades(trades)
    print(f"Upsert complete. Inserted {inserted} new trades (scraped {len(trades)} total).")
    
    # --- UPDATE ALL HISTORICAL PRICES ---
    # Now that we've inserted new trades, update current prices for ALL
    # existing trades in the database so the dashboard reflects live profits.
    print("Updating current prices for all historical trades in the database...")
    update_all_current_prices()
    
    # --- ENRICH TICKER METADATA (SECTORS / INDUSTRIES) ---
    print("Enriching ticker metadata (sectors & industries)...")
    enriched_count = enrich_ticker_metadata(max_tickers=100)
    print(f"Enriched metadata for {enriched_count} new tickers.")


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
