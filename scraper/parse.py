"""Parsing helpers for Senate eFD report search and (later) trades.

This module provides:

* Functions to turn raw report rows from the search API into
  structured dicts / DataFrames.
* Helpers to interpret amount ranges and normalize transaction types
  (used later when parsing individual PTR pages).
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Iterable, List, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from .session import BASE_URL


def parse_report_row(row: list[str]) -> dict:
    """Parse a single report row from the search API into a dict.

    Raw ``row`` format from the API is::

        [first_name, last_name, role, html_link, filed_date_str]

    Example::

        ['John', 'Boozman', 'Boozman, John (Senator)',
         '<a href="/search/view/ptr/.../" target="_blank">Periodic Transaction Report for 01/13/2026</a>',
         '01/13/2026']
    """

    if len(row) < 5:
        raise ValueError(f"Unexpected report row length {len(row)}: {row!r}")

    first_name, last_name, role, link_html, filed_str = row[:5]

    # Parse the <a> tag to extract href and link text
    soup = BeautifulSoup(link_html, "html.parser")
    a = soup.find("a")
    if a is None or not a.get("href"):
        raise ValueError(f"Could not find <a> tag with href in link_html={link_html!r}")

    report_path = a["href"]
    report_title = a.get_text(strip=True)

    if report_path.startswith("http://") or report_path.startswith("https://"):
        report_url = report_path
    else:
        report_url = f"{BASE_URL}{report_path}"

    # Path is expected to look like: /search/view/{format}/{uuid}/
    report_format = None
    report_id = None
    path_parts = report_path.strip("/").split("/")
    if len(path_parts) >= 4 and path_parts[0] == "search" and path_parts[1] == "view":
        report_format = path_parts[2]
        report_id = path_parts[3]

    # Derive a simplified report_type from the link text
    title_lower = report_title.lower()
    if "periodic transaction report" in title_lower:
        report_type = "PTR"
    elif "annual report" in title_lower:
        report_type = "Annual"
    elif "extension" in title_lower:
        report_type = "Extension"
    else:
        report_type = "Other"

    # Filing date
    filing_date = dt.datetime.strptime(filed_str, "%m/%d/%Y").date()

    return {
        "senator_first_name": first_name.strip(),
        "senator_last_name": last_name.strip(),
        "senator_display_name": role.strip(),
        "chamber": "Senate",
        "report_type": report_type,
        "report_format": report_format,
        "report_id": report_id,
        "report_url": report_url,
        "report_path": report_path,
        "report_title": report_title,
        "filing_date": filing_date,
        "raw_link_html": link_html,
        "is_ptr": report_type == "PTR",
    }


def parse_report_rows(rows: Iterable[list[str]]) -> List[dict]:
    """Parse many report rows from the search API into a list of dicts."""

    return [parse_report_row(row) for row in rows]


def reports_to_dataframe(reports: Iterable[dict]) -> pd.DataFrame:
    """Convert an iterable of report dicts into a pandas DataFrame."""

    df = pd.DataFrame(list(reports))
    # Optional: enforce a column order that is convenient to work with
    preferred_order = [
        "filing_date",
        "report_type",
        "report_format",
        "report_id",
        "report_title",
        "report_url",
        "senator_display_name",
        "senator_first_name",
        "senator_last_name",
        "chamber",
        "is_ptr",
    ]
    existing_cols = [c for c in preferred_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    return df[existing_cols + other_cols]


def parse_amount_range(amount_str: str | None) -> Tuple[float | None, float | None, float | None]:
    """Parse an amount range string into (min, max, midpoint).

    Examples of inputs (exact formats may vary on PTR pages)::

        "$1,001 - $15,000" -> (1001.0, 15000.0, 8000.5)
        "$15,001 - $50,000" -> (...)
        "Over $1,000,000"    -> (1000000.0, None, None)

    This is a utility for later when parsing individual PTR trades.
    """

    if not amount_str:
        return None, None, None

    s = amount_str.strip()
    # Handle "Over $X" style
    if s.lower().startswith("over"):
        # e.g. "Over $1,000,000"
        num_part = s[4:].strip()  # remove "Over"
        num_part = num_part.replace("$", "").replace(",", "").strip()
        try:
            min_val = float(num_part)
        except ValueError:
            return None, None, None
        return min_val, None, None

    # Handle ranges like "$1,001 - $15,000"
    # There might be different separators; we focus on "-" for now.
    if "-" in s:
        low_str, high_str = s.split("-", 1)
        low_str = low_str.replace("$", "").replace(",", "").strip()
        high_str = high_str.replace("$", "").replace(",", "").strip()
        try:
            low = float(low_str)
            high = float(high_str)
        except ValueError:
            return None, None, None
        mid = (low + high) / 2.0
        return low, high, mid

    # Fallback: try to parse as a single numeric amount
    s_clean = s.replace("$", "").replace(",", "").strip()
    try:
        val = float(s_clean)
    except ValueError:
        return None, None, None
    return val, val, val


def normalize_transaction_type(raw: str | None) -> str | None:
    """Normalize a raw transaction type string to a small canonical set.

    The exact values will depend on how PTR tables label transactions,
    but a typical mapping might be::

        "Purchase"            -> "BUY"
        "Sale (Full)"         -> "SELL"
        "Sale (Partial)"      -> "SELL"
        "Exchange"            -> "EXCHANGE"
    """

    if raw is None:
        return None

    text = raw.strip().lower()
    if not text:
        return None

    if "purchase" in text or text == "p":
        return "BUY"
    if "sale" in text or text == "s":
        return "SELL"
    if "exchange" in text:
        return "EXCHANGE"

    return raw.strip()
