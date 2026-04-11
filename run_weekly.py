"""Weekly ingest entry point.

Runs both pipelines:
  1. Senate PTR scraper for the last 7 days (catches any delayed filings
     from the past week that the daily run may have missed).
  2. Full RapidAPI Congress pull for House + Senate historical trades.

Called by the GitHub Actions weekly cron
(.github/workflows/weekly_scrape.yml).

Usage:
    python run_weekly.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from ingest.senate_ptr import run_ingest as senate_ingest
from ingest.congress_api import run_ingest as congress_ingest
from ingest.portfolio_snapshots import run_snapshot_ingest
from ingest.lobbying import run_ingest as lobbying_ingest
from ingest.gov_contracts import run_ingest as contracts_ingest
from ingest.activist_filings import run_ingest as activist_ingest


def main() -> None:
    print("=== Weekly ingest: Senate PTR (last 7 days) ===")
    senate_ingest(days=7)

    print("\n=== Weekly ingest: Congress API (full history) ===")
    congress_ingest()

    print("\n=== Weekly ingest: Portfolio snapshots ===")
    run_snapshot_ingest()

    print("\n=== Weekly ingest: Corporate lobbying (last 120 days) ===")
    lobbying_ingest(days=120)

    print("\n=== Weekly ingest: Government contracts (last 90 days) ===")
    contracts_ingest(days=90)

    print("\n=== Weekly ingest: Activist 13D/G filings (last 90 days) ===")
    activist_ingest(days=90)


if __name__ == "__main__":
    main()
