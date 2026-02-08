"""Fetch and parse individual Periodic Transaction Reports (PTRs).

Given a parsed report summary (from ``scraper.parse.parse_report_row``),
this module fetches the PTR HTML page and extracts the Transactions
table into a list of normalized trade records.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

import pandas as pd
from bs4 import BeautifulSoup

from .session import create_efd_session
from .parse import parse_amount_range, normalize_transaction_type


def fetch_report_html(report_url: str, session=None) -> str:
    """Fetch the HTML for a single report URL using an authenticated session.

    Adds debug logging so we can see whether we're actually getting the PTR
    detail page or being bounced back to the generic home/search page.
    """

    # Allow caller to reuse an existing authenticated session for efficiency
    # and to better mirror real browser behaviour across multiple requests.
    if session is None:
        session, _ = create_efd_session()
    resp = session.get(report_url, allow_redirects=True)

    # Basic debug about what we actually received
    print(f"DEBUG: PTR GET status={resp.status_code}, final_url={resp.url}")

    soup = BeautifulSoup(resp.text, "html.parser")
    title = soup.title.string if soup.title else "<no title>"
    print(f"DEBUG: HTML Title is: {title}")

    resp.raise_for_status()
    return resp.text


def _find_transactions_table(soup: BeautifulSoup) -> Any:
    """Locate the Transactions table element in a PTR HTML document.

    The HTML you provided shows a structure like::

        <section class="card">
          ...
          <div class="table-responsive">
            <table class="table table-striped">
              <caption>List of transactions added to this report</caption>
              <thead> ... </thead>
              <tbody> ... </tbody>

    We search for a <table> whose caption mentions "transactions".
    """

    for table in soup.find_all("table"):
        caption = table.find("caption")
        if caption and "transaction" in caption.get_text(strip=True).lower():
            return table

    # Fallback: first striped table
    table = soup.find("table", class_="table")
    if table is None:
        raise ValueError("Could not find transactions table in PTR HTML")
    return table


def parse_ptr_trades_from_html(html: str, report_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse trades from a PTR HTML page.

    ``report_meta`` is a dict produced by ``parse_report_row`` and should
    contain keys like ``senator_first_name``, ``senator_last_name``,
    ``senator_display_name``, ``report_id``, ``report_type``, and
    ``filing_date``.
    """

    soup = BeautifulSoup(html, "html.parser")
    table = _find_transactions_table(soup)

    tbody = table.find("tbody")
    if tbody is None:
        return []

    trades: List[Dict[str, Any]] = []

    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 9:
            continue

        # Columns:
        # 0: index
        # 1: Transaction Date
        # 2: Owner
        # 3: Ticker (may contain <a>)
        # 4: Asset Name
        # 5: Asset Type
        # 6: Type (e.g. "Sale (Full)")
        # 7: Amount (range string)
        # 8: Comment

        transaction_date_raw = tds[1].get_text(strip=True)
        owner = tds[2].get_text(strip=True) or None

        ticker_td = tds[3]
        ticker_link = ticker_td.find("a")
        ticker = (
            ticker_link.get_text(strip=True)
            if ticker_link is not None
            else ticker_td.get_text(strip=True) or None
        )
        private_tickers = {"", "-", "--"}
        if ticker is None or ticker in private_tickers:
            # Skip this row entirely – non-public or unidentifiable asset
            continue

        asset_name = tds[4].get_text(strip=True) or None
        asset_type = tds[5].get_text(strip=True) or None
        raw_tx_type = tds[6].get_text(strip=True)
        amount_range_raw = tds[7].get_text(strip=True) or None
        comment_raw = tds[8].get_text(strip=True)
        comment = None if comment_raw == "--" or comment_raw == "" else comment_raw

        # Parse date and amount
        try:
            transaction_date = dt.datetime.strptime(transaction_date_raw, "%m/%d/%Y").date()
        except ValueError:
            # If parsing fails, store raw string and leave date None
            transaction_date = None

        amount_min, amount_max, mid_point = parse_amount_range(amount_range_raw)
        transaction_type = normalize_transaction_type(raw_tx_type)

        senator_first = report_meta.get("senator_first_name") or ""
        senator_last = report_meta.get("senator_last_name") or ""
        senator_name = f"{senator_first} {senator_last}".strip()

        trade: Dict[str, Any] = {
            "senator_name": senator_name,
            "senator_first_name": senator_first,
            "senator_last_name": senator_last,
            "senator_display_name": report_meta.get("senator_display_name"),
            "chamber": report_meta.get("chamber", "Senate"),
            "report_id": report_meta.get("report_id"),
            "report_type": report_meta.get("report_type"),
            "report_format": report_meta.get("report_format"),
            "filing_date": report_meta.get("filing_date"),
            "transaction_date": transaction_date,
            "owner": owner,
            "ticker": ticker,
            "asset_name": asset_name,
            "asset_type": asset_type,
            "transaction_type": transaction_type,
            "transaction_type_raw": raw_tx_type,
            "amount_range_raw": amount_range_raw,
            "amount_min": amount_min,
            "amount_max": amount_max,
            "mid_point": mid_point,
            "comment": comment,
        }

        trades.append(trade)

    return trades


def fetch_ptr_trades(report_meta: Dict[str, Any], session=None) -> List[Dict[str, Any]]:
    report_url = report_meta["report_url"]
    html = fetch_report_html(report_url, session=session)
    
    # DEBUG: Check if we actually got the report or just the landing page
    if "Agreement" in html or "Prohibited" in html:
        print("DEBUG: Caught by the landing/disclaimer page!")
        
    return parse_ptr_trades_from_html(html, report_meta)


def trades_to_dataframe(trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of trade dicts to a pandas DataFrame."""

    return pd.DataFrame(trades)
