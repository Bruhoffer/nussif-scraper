"""Backfill Senate PTR trades for the last N years.

A thin wrapper around the Senate PTR ingest pipeline that simply
requests a long lookback window. The ingest is idempotent — existing
trades are skipped — so this can be safely rerun.

Usage:
    python backfill.py --years 5
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

load_dotenv()

from ingest.senate_ptr import run_ingest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill Senate PTR trades for the last N years"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="How many years back from today to fetch filings (default: 2)",
    )
    args = parser.parse_args()

    days = args.years * 365
    print(f"[backfill] Starting backfill for last {args.years} years (~{days} days)...")
    run_ingest(days=days)
    print("[backfill] Backfill complete.")


if __name__ == "__main__":
    main()
