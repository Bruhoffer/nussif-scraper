"""Helpers to load PTR trades from the local database into DataFrames."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from ..db.config import engine


def load_trades_df(days: int = 90) -> pd.DataFrame:
    """Load trades from the DB for the last ``days`` days of filings.

    This filters on ``filing_date`` so the dashboard reflects recent
    disclosure activity.
    """

    query = text(
        """
        SELECT *
        FROM trades
        WHERE filing_date >= date('now', :offset)
        ORDER BY filing_date DESC
        """
    )

    offset = f"-{days} days"
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"offset": offset})

