"""Quick sanity-check script for parsing trades from a PTR.

Run with your conda env activated from the project root:

    conda activate insiderenv
    cd "insiderscraper"
    python test_ptr_trades.py
"""

from __future__ import annotations

import datetime as dt

from scraper.fetch import fetch_all_reports
from scraper.parse import parse_report_rows
from scraper.ptr_details import fetch_ptr_trades


def main() -> None:
    today = dt.date.today()
    start_date = today - dt.timedelta(days=30)

    print(f"Fetching reports filed between {start_date} and {today}...")
    result = fetch_all_reports(submitted_start_date=start_date, submitted_end_date=today)

    rows = result.get("data", [])
    reports = parse_report_rows(rows)

    # Pick the first PTR report
    ptr_reports = [
        r for r in reports
        if r.get("is_ptr") and r.get("report_format") == "ptr"
    ]
    if not ptr_reports:
        print("No PTR reports found in this date range.")
        return

    report = ptr_reports[0]
    print(
        "Using PTR:",
        report["filing_date"],
        "|",
        report["senator_display_name"],
        "|",
        report["report_url"],
    )

    trades = fetch_ptr_trades(report)
    print(f"\nFound {len(trades)} trades in this PTR. Sample:")
    for t in trades[:5]:
        print(
            f"{t['transaction_date']} | {t['senator_display_name']} | "
            f"{t['ticker']} | {t['asset_name']} | {t['transaction_type']} | {t['amount_range_raw']}"
        )


if __name__ == "__main__":
    main()
