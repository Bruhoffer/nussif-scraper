"""CLI utility to fetch PTR trades over a date range and dump to CSV.

Usage (from project root, with env activated):

    cd "insiderscraper"
    python ptr_batch_to_csv.py --days 90 --out data/ptr_trades_90d.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from scraper.pipeline import fetch_ptr_trades_for_range


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PTR trades and write to CSV")
    parser.add_argument("--days", type=int, default=90, help="How many days back from today to fetch filings")
    parser.add_argument("--out", type=str, default="data/ptr_trades.csv", help="Output CSV filepath")
    args = parser.parse_args()

    today = dt.date.today()
    start_date = today - dt.timedelta(days=args.days)

    print(f"Fetching PTR trades filed between {start_date} and {today}...")
    df = fetch_ptr_trades_for_range(start_date, today)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("No PTR trades found in this range. Nothing written.")
        return

    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} trades to {out_path}")


if __name__ == "__main__":
    main()
