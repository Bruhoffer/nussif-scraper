from __future__ import annotations

from datetime import date, timedelta

import time
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

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

    query = text(
        """
        SELECT *
        FROM trades
        WHERE filing_date >= :cutoff
        ORDER BY filing_date DESC
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

