"""Shared post-processing logic used by all ingest pipelines.

Both senate_ptr.py and congress_api.py run the same sequence after
fetching raw trades:
  1. Enrich with historical + current prices (via yfinance / PriceCache)
  2. Clean NaN / inf values so the DB driver receives plain Python types
  3. Upsert into the trades table (idempotent)

This module centralises those steps so each pipeline only has to call
clean_and_upsert(df).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from db.prices import enrich_prices_for_trades
from db.upsert import upsert_trades


def clean_and_upsert(df: pd.DataFrame) -> int:
    """Enrich prices, clean a trades DataFrame, and upsert to the DB.

    Parameters
    ----------
    df:
        DataFrame whose columns match the Trade ORM model. Must include
        at least 'ticker' and 'transaction_date' for price enrichment.

    Returns
    -------
    int
        Number of newly inserted rows.
    """

    if df.empty:
        return 0

    # 1. Enrich with price_at_transaction and current_price
    failed_tickers, failed_pairs = enrich_prices_for_trades(df)
    if failed_tickers or failed_pairs:
        print(
            f"[ingest] Pricing missing for {len(failed_tickers)} tickers "
            f"and {len(failed_pairs)} (ticker, date) pairs — stored as NULL."
        )

    # 2. Round price columns to 4 decimal places
    for col in ("price_at_transaction", "current_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4)

    # 3. Replace all NaN / inf variants with None (SQL NULL)
    df = df.astype(object).replace({np.nan: None, float("inf"): None, float("-inf"): None})
    df = df.where(pd.notnull(df), None)

    # 4. Upsert
    inserted = upsert_trades(df.to_dict(orient="records"))
    return inserted
