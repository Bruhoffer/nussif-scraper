from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text

from db.config import engine


def load_trades_df(days: int = 90) -> pd.DataFrame:
    """Load trades from the DB for the last ``days`` days of filings.

    This filters on ``filing_date`` so the dashboard reflects recent
    disclosure activity.

    Uses a DB-agnostic comparison on a concrete cutoff date instead of
    SQLite-specific ``date('now', ...)`` syntax, so it works on both
    SQLite and Azure SQL.
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

    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"cutoff": cutoff})

