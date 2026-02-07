"""Higher-level scraping pipeline for Senate PTR trades.

This module stitches together the lower-level report search and PTR
parsing logic to fetch all PTR trades in a given filing date range.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Iterable, List

import pandas as pd

from .fetch import fetch_all_reports
from .parse import parse_report_rows
from .ptr_details import fetch_ptr_trades
from .session import create_efd_session


def fetch_ptr_reports_for_range(start_date: dt.date, end_date: dt.date) -> List[Dict[str, Any]]:
    """Fetch all PTR report metadata for filings in [start_date, end_date]."""

    session, _ = create_efd_session()
    result = fetch_all_reports(
        submitted_start_date=start_date,
        submitted_end_date=end_date,
        session=session,
    )

    rows = result.get("data", [])
    reports = parse_report_rows(rows)

    ptr_reports = [
        r for r in reports
        if r.get("is_ptr") and r.get("report_format") == "ptr"
    ]
    return ptr_reports


def fetch_ptr_trades_for_reports(reports: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fetch and parse trades for a collection of PTR reports."""

    session, _ = create_efd_session()
    all_trades: List[Dict[str, Any]] = []

    for report in reports:
        trades = fetch_ptr_trades(report, session=session)
        all_trades.extend(trades)

    return all_trades


def fetch_ptr_trades_for_range(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Fetch all PTR trades for filings in [start_date, end_date] as a DataFrame."""

    reports = fetch_ptr_reports_for_range(start_date, end_date)
    trades = fetch_ptr_trades_for_reports(reports)
    return pd.DataFrame(trades)
