from __future__ import annotations

from datetime import date, timedelta

import time
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

import sys
import os
# Add the parent directory (project root) to sys.path so we can import 'db'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from db.config import engine


def load_trades_df(
    days: int = 90,
    max_retries: int = 5,
    backoff_seconds: float = 2.0,
) -> pd.DataFrame:
    """Load trades from the DB for the last ``days`` days of filings.

    This filters on ``filing_date`` so the dashboard reflects recent
    disclosure activity.

    Uses a DB-agnostic comparison on a concrete cutoff date instead of
    SQLite-specific ``date('now', ...)`` syntax, so it works on both
    SQLite and Azure SQL.

    Includes simple retry logic for transient connection timeouts
    (e.g. HYT00 / "Login timeout expired").
    """

    cutoff = date.today() - timedelta(days=days)

    # Join ticker_metadata so downstream views can use real sector/
    # industry information instead of placeholder values.
    query = text(
        """
        SELECT t.*, m.company_name, m.sector, m.industry
        FROM trades AS t
        LEFT JOIN ticker_metadata AS m
          ON t.ticker = m.ticker
        WHERE t.filing_date >= :cutoff
        ORDER BY t.filing_date DESC
        """
    )

    for attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                return pd.read_sql(query, conn, params={"cutoff": cutoff})

        except OperationalError as exc:
            # Detect the specific timeout pattern conservatively; for
            # non-timeout OperationalErrors, or after exhausting retries,
            # we re-raise immediately.
            msg = str(exc.orig) if getattr(exc, "orig", None) else str(exc)
            is_timeout = "HYT00" in msg or "Login timeout expired" in msg

            if not is_timeout or attempt == max_retries:
                raise

            sleep_for = backoff_seconds * attempt
            print(
                f"[load_trades_df] Transient DB timeout, "
                f"retry {attempt}/{max_retries} in {sleep_for:.1f}s: {msg}"
            )
            time.sleep(sleep_for)

    # If we somehow fall through without returning or raising, just
    # return an empty DataFrame as a very defensive fallback.
    return pd.DataFrame()


def load_all_trades_df() -> pd.DataFrame:
    """Load every trade in the DB with no date filter, joining ticker_metadata.

    Used for cross-year analysis like current holdings and portfolio curves
    where filtering to 365 days would give incorrect results (e.g. a 2022
    buy that was never sold would appear as an open position if we only see
    the last year).
    """

    query = text(
        """
        SELECT t.*, m.company_name, m.sector, m.industry
        FROM trades AS t
        LEFT JOIN ticker_metadata AS m
          ON t.ticker = m.ticker
        WHERE t.transaction_date IS NOT NULL
        ORDER BY t.transaction_date ASC
        """
    )

    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except (OperationalError, SQLAlchemyError) as exc:
        print(f"[load_all_trades_df] DB query failed: {exc}")
        return pd.DataFrame()


def load_volume_by_year_df() -> pd.DataFrame:
    """Load the minimal columns needed to build the trade volume by year chart.

    Returns all trades in the DB (no date filter) with only the columns
    required for year-level aggregation: transaction_date, transaction_type,
    mid_point, chamber, senator_display_name.  Kept lightweight intentionally
    so it can be cached separately from the main 365-day dataset.
    """

    query = text(
        """
        SELECT transaction_date, transaction_type, mid_point,
               chamber, senator_display_name
        FROM trades
        WHERE transaction_date IS NOT NULL
          AND mid_point IS NOT NULL
        ORDER BY transaction_date ASC
        """
    )

    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except (OperationalError, SQLAlchemyError) as exc:
        print(f"[load_volume_by_year_df] DB query failed: {exc}")
        return pd.DataFrame()


def load_portfolio_curve(senator_display_name: str) -> pd.DataFrame:
    """Load the precomputed portfolio curve for a senator from the DB.

    Returns a DataFrame with columns: date, portfolio_value.
    Returns an empty DataFrame if no snapshots exist yet for this senator
    (first run before ingest/portfolio_snapshots.py has been executed).
    """

    query = text(
        """
        SELECT snapshot_date AS date, portfolio_value
        FROM portfolio_snapshots
        WHERE senator_display_name = :senator
        ORDER BY snapshot_date ASC
        """
    )

    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"senator": senator_display_name})
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"]).dt.date
            return df
    except (OperationalError, SQLAlchemyError) as exc:
        print(f"[load_portfolio_curve] DB query failed: {exc}")
        return pd.DataFrame()


def load_all_portfolio_curves() -> pd.DataFrame:
    """Load portfolio snapshots for all senators in one query.

    Returns a DataFrame with columns: senator_display_name, date, portfolio_value.
    Used for computing cross-senator metrics like Sharpe ratio leaderboard.
    """
    query = text(
        """
        SELECT senator_display_name, snapshot_date AS date, portfolio_value
        FROM portfolio_snapshots
        ORDER BY senator_display_name, snapshot_date ASC
        """
    )
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
            return df
    except (OperationalError, SQLAlchemyError) as exc:
        print(f"[load_all_portfolio_curves] DB query failed: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Market Intelligence loaders
# ---------------------------------------------------------------------------

def load_lobbying_df(days_back: int = 120) -> pd.DataFrame:
    """Load lobbying filings from the last ``days_back`` days."""
    cutoff = date.today() - timedelta(days=days_back)
    query = text("""
        SELECT filing_uuid, client_name, ticker, registrant_name,
               amount, filing_date, period_of_lobbying, specific_issues, source_url
        FROM lobbying_filings
        WHERE filing_date >= :cutoff
        ORDER BY filing_date DESC
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"cutoff": cutoff})
    except Exception as exc:
        print(f"[load_lobbying_df] DB query failed: {exc}")
        return pd.DataFrame()


def load_lobbying_top_spenders(days_back: int = 120) -> pd.DataFrame:
    """Return top lobbying spenders grouped by client/ticker, summed amount."""
    cutoff = date.today() - timedelta(days=days_back)
    query = text("""
        SELECT
            COALESCE(ticker, client_name) AS label,
            client_name,
            ticker,
            SUM(amount) AS total_amount,
            COUNT(*) AS filings
        FROM lobbying_filings
        WHERE filing_date >= :cutoff AND amount IS NOT NULL
        GROUP BY COALESCE(ticker, client_name), client_name, ticker
        ORDER BY total_amount DESC
        LIMIT 20
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"cutoff": cutoff})
    except Exception as exc:
        print(f"[load_lobbying_top_spenders] DB query failed: {exc}")
        return pd.DataFrame()


def load_gov_contracts_df(days_back: int = 90) -> pd.DataFrame:
    """Load government contract awards from the last ``days_back`` days."""
    cutoff = date.today() - timedelta(days=days_back)
    query = text("""
        SELECT award_id, recipient_name, ticker, award_amount,
               award_date, funding_agency, description
        FROM gov_contracts
        WHERE award_date >= :cutoff
        ORDER BY award_date DESC
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"cutoff": cutoff})
    except Exception as exc:
        print(f"[load_gov_contracts_df] DB query failed: {exc}")
        return pd.DataFrame()


def load_gov_contracts_top_recipients(days_back: int = 90) -> pd.DataFrame:
    """Return top contract recipients grouped by ticker/company, summed amount."""
    cutoff = date.today() - timedelta(days=days_back)
    query = text("""
        SELECT
            COALESCE(ticker, recipient_name) AS label,
            recipient_name,
            ticker,
            SUM(award_amount) AS total_amount,
            COUNT(*) AS contracts
        FROM gov_contracts
        WHERE award_date >= :cutoff AND award_amount IS NOT NULL
        GROUP BY COALESCE(ticker, recipient_name), recipient_name, ticker
        ORDER BY total_amount DESC
        LIMIT 20
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"cutoff": cutoff})
    except Exception as exc:
        print(f"[load_gov_contracts_top_recipients] DB query failed: {exc}")
        return pd.DataFrame()


def load_activist_filings_df(days_back: int = 90) -> pd.DataFrame:
    """Load 13D/G activist filings from the last ``days_back`` days."""
    cutoff = date.today() - timedelta(days=days_back)
    query = text("""
        SELECT accession_number, ticker, target_company, lead_investor,
               form_type, filing_date, shares, prev_shares,
               ownership_pct, shares_change_pct, market_cap_m, sec_url
        FROM activist_filings
        WHERE filing_date >= :cutoff
        ORDER BY filing_date DESC
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"cutoff": cutoff})
    except Exception as exc:
        print(f"[load_activist_filings_df] DB query failed: {exc}")
        return pd.DataFrame()


def load_market_intelligence_overlap() -> pd.DataFrame:
    """Return tickers that appear in both congress trades and any MI table.

    Columns: ticker, senator_display_name, transaction_date, transaction_type,
             mid_point, in_lobbying, in_contracts, in_activist.
    """
    query = text("""
        SELECT
            t.ticker,
            t.senator_display_name,
            t.transaction_date,
            t.transaction_type,
            t.mid_point,
            CASE WHEN l.ticker IS NOT NULL THEN 1 ELSE 0 END AS in_lobbying,
            CASE WHEN c.ticker IS NOT NULL THEN 1 ELSE 0 END AS in_contracts,
            CASE WHEN a.ticker IS NOT NULL THEN 1 ELSE 0 END AS in_activist
        FROM trades t
        LEFT JOIN (SELECT DISTINCT ticker FROM lobbying_filings  WHERE ticker IS NOT NULL) l ON t.ticker = l.ticker
        LEFT JOIN (SELECT DISTINCT ticker FROM gov_contracts      WHERE ticker IS NOT NULL) c ON t.ticker = c.ticker
        LEFT JOIN (SELECT DISTINCT ticker FROM activist_filings   WHERE ticker IS NOT NULL) a ON t.ticker = a.ticker
        WHERE (l.ticker IS NOT NULL OR c.ticker IS NOT NULL OR a.ticker IS NOT NULL)
          AND t.transaction_date IS NOT NULL
        ORDER BY t.transaction_date DESC
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn)
    except Exception as exc:
        print(f"[load_market_intelligence_overlap] DB query failed: {exc}")
        return pd.DataFrame()


def load_ticker_timeline(ticker: str) -> dict[str, pd.DataFrame]:
    """Load all event data for one ticker for the cross-reference timeline.

    Returns a dict with keys: 'prices', 'trades', 'contracts', 'lobbying', 'filings'.
    Each value is a DataFrame with at least a 'date' column.
    """
    result: dict[str, pd.DataFrame] = {}

    def _q(query_str, params=None):
        try:
            with engine.connect() as conn:
                return pd.read_sql(text(query_str), conn, params=params or {})
        except Exception as exc:
            print(f"[load_ticker_timeline] Query failed: {exc}")
            return pd.DataFrame()

    result["prices"] = _q("""
        SELECT date, price FROM price_cache
        WHERE ticker = :ticker ORDER BY date ASC
    """, {"ticker": ticker})

    result["trades"] = _q("""
        SELECT transaction_date AS date, senator_display_name,
               transaction_type, mid_point
        FROM trades WHERE ticker = :ticker
          AND transaction_date IS NOT NULL
        ORDER BY transaction_date ASC
    """, {"ticker": ticker})

    result["contracts"] = _q("""
        SELECT award_date AS date, recipient_name, award_amount, funding_agency
        FROM gov_contracts WHERE ticker = :ticker
          AND award_date IS NOT NULL
        ORDER BY award_date ASC
    """, {"ticker": ticker})

    result["lobbying"] = _q("""
        SELECT filing_date AS date, client_name, amount, period_of_lobbying
        FROM lobbying_filings WHERE ticker = :ticker
          AND filing_date IS NOT NULL
        ORDER BY filing_date ASC
    """, {"ticker": ticker})

    result["filings"] = _q("""
        SELECT filing_date AS date, lead_investor, form_type, ownership_pct, sec_url
        FROM activist_filings WHERE ticker = :ticker
          AND filing_date IS NOT NULL
        ORDER BY filing_date ASC
    """, {"ticker": ticker})

    return result


def warm_up_db() -> None:
    """Run a lightweight query to warm up the DB connection.

    Best-effort only; errors are logged but not raised so that app
    startup is never blocked by this warm-up.
    """

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except (OperationalError, SQLAlchemyError) as exc:
        print(f"[warm_up_db] DB warm-up failed (ignored): {exc}")

