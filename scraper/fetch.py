"""Low-level fetch functions for the Senate eFD search API."""

from __future__ import annotations
import datetime as dt
from typing import Any, Dict
from .session import BASE_URL, create_efd_session, default_ajax_headers

REPORT_DATA_URL = f"{BASE_URL}/search/report/data/"

def _format_datetime_for_query(date: dt.date) -> str:
    """Format: MM/DD/YYYY 00:00:00"""
    return date.strftime("%m/%d/%Y 00:00:00")

def _base_datatables_payload() -> Dict[str, Any]:
    """Baseline payload mirroring the browser's DataTables request."""
    return {
        "draw": "1",
        "columns[0][data]": "0",
        "columns[0][name]": "",
        "columns[0][searchable]": "true",
        "columns[0][orderable]": "true",
        "columns[0][search][value]": "",
        "columns[0][search][regex]": "false",
        "columns[1][data]": "1",
        "columns[1][name]": "",
        "columns[1][searchable]": "true",
        "columns[1][orderable]": "true",
        "columns[1][search][value]": "",
        "columns[1][search][regex]": "false",
        "columns[2][data]": "2",
        "columns[2][name]": "",
        "columns[2][searchable]": "true",
        "columns[2][orderable]": "true",
        "columns[2][search][value]": "",
        "columns[2][search][regex]": "false",
        "columns[3][data]": "3",
        "columns[3][name]": "",
        "columns[3][searchable]": "true",
        "columns[3][orderable]": "true",
        "columns[3][search][value]": "",
        "columns[3][search][regex]": "false",
        "columns[4][data]": "4",
        "columns[4][name]": "",
        "columns[4][searchable]": "true",
        "columns[4][orderable]": "true",
        "columns[4][search][value]": "",
        "columns[4][search][regex]": "false",
        "order[0][column]": "1",
        "order[0][dir]": "asc",
        "order[1][column]": "0",
        "order[1][dir]": "asc",
        "search[value]": "",
        "search[regex]": "false",
        "report_types": "[]",
        "filer_types": "[]",
        "candidate_state": "",
        "senator_state": "",
        "office_id": "",
    }

def fetch_reports_page(
    submitted_start_date: dt.date,
    submitted_end_date: dt.date | None = None,
    first_name: str = "",
    last_name: str = "",
    start: int = 0,
    length: int = 100,
    session=None,
) -> Dict[str, Any]:
    """Fetch a single page of reports.

    If ``session`` is provided, it will be reused (allowing callers to
    share a single authenticated session across multiple requests).
    Otherwise, a new eFD session is created for this call.
    """

    # 1. Get the session and the CSRF token from the cookie
    if session is None:
        session, _ = create_efd_session()
    csrf = session.cookies.get("csrftoken")
    
    # 2. Build headers with that token
    headers = default_ajax_headers(csrf)

    # 3. Build payload
    payload = _base_datatables_payload()
    payload.update({
        "start": str(start),
        "length": str(length),
        "submitted_start_date": _format_datetime_for_query(submitted_start_date),
        "submitted_end_date": (
            _format_datetime_for_query(submitted_end_date)
            if submitted_end_date is not None else ""
        ),
        "first_name": first_name,
        "last_name": last_name,
    })

    # 4. Execute POST
    resp = session.post(REPORT_DATA_URL, data=payload, headers=headers)
    
    if resp.status_code == 403:
        print("CRITICAL: Still getting 403. Check if CSRF token is being passed correctly in cookies.")
        
    resp.raise_for_status()
    return resp.json()

def fetch_all_reports(
    submitted_start_date: dt.date,
    submitted_end_date: dt.date | None = None,
    first_name: str = "",
    last_name: str = "",
    page_size: int = 100,
    session=None,
) -> Dict[str, Any]:
    """Fetch all matching reports, transparently handling pagination.

    Returns a dict with at least:

    - ``recordsTotal``
    - ``recordsFiltered``
    - ``data``: combined list of all rows from every page
    """

    # Ensure we have a session to reuse across all pages
    if session is None:
        session, _ = create_efd_session()

    first_page = fetch_reports_page(
        submitted_start_date=submitted_start_date,
        submitted_end_date=submitted_end_date,
        first_name=first_name,
        last_name=last_name,
        start=0,
        length=page_size,
        session=session,
    )

    data = list(first_page.get("data", []))
    records_filtered = int(first_page.get("recordsFiltered", len(data)))

    # If everything fit on the first page, we're done.
    if len(data) >= records_filtered:
        return {
            "recordsTotal": first_page.get("recordsTotal", len(data)),
            "recordsFiltered": records_filtered,
            "data": data,
        }

    # Otherwise, keep pulling more pages until we've got them all.
    start = page_size
    while len(data) < records_filtered:
        page = fetch_reports_page(
            submitted_start_date=submitted_start_date,
            submitted_end_date=submitted_end_date,
            first_name=first_name,
            last_name=last_name,
            start=start,
            length=page_size,
            session=session,
        )
        rows = page.get("data", [])
        if not rows:
            break
        data.extend(rows)
        start += page_size

    return {
        "recordsTotal": first_page.get("recordsTotal", len(data)),
        "recordsFiltered": records_filtered,
        "data": data,
    }
