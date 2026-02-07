"""Quick sanity-check script for the Senate eFD report search.

Run this with your conda env activated from the project root:

    conda activate insiderenv
    cd "insiderscraper"
    python test_report_search.py
"""

from __future__ import annotations

import datetime as dt

from scraper.fetch import fetch_all_reports
from scraper.parse import parse_report_rows


def main() -> None:
    today = dt.date.today()
    # For now, just look back 30 days of filings overall
    start_date = today - dt.timedelta(days=30)

    print(f"Fetching reports filed between {start_date} and {today}...")
    result = fetch_all_reports(submitted_start_date=start_date, submitted_end_date=today)

    rows = result.get("data", [])
    reports = parse_report_rows(rows)

    print(f"recordsFiltered = {result.get('recordsFiltered')}\n")
    print("First few parsed reports:")
    for r in reports[:5]:
        print(
            f"{r['filing_date']} | {r['senator_display_name']} | "
            f"{r['report_type']} | {r['report_url']}"
        )


if __name__ == "__main__":
    main()
