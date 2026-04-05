"""Daily ingest entry point.

Runs the Senate PTR scraper for the last 3 days. Called by the
GitHub Actions daily cron (.github/workflows/daily_scrape.yml).

Usage:
    python run_daily.py
    python run_daily.py --days 7   # optional override
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from ingest.senate_ptr import run_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Senate PTR ingest")
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="How many days back from today to fetch filings (default: 3)",
    )
    args = parser.parse_args()
    run_ingest(days=args.days)


if __name__ == "__main__":
    main()
