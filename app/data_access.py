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

