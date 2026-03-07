"""Analysis helpers for senator PTR trade backtests.

This module provides a small set of utilities that your notebooks can
import to:

* Load trades from the database over a given window.
* Prepare trade-level features (direction, quarters, etc.).
* Compute direction-adjusted returns and PnL using the existing
  ``price_at_transaction`` and ``current_price`` columns.
* Optionally compute multi-horizon close-to-close returns using the
  shared PriceCache / yfinance integration.
* Aggregate by senator, quarter, ticker, and compute quartile-based
  "suspiciousness" metrics (share of top-quartile trades).

These helpers are intentionally generic and opinionated so that the
notebooks can stay relatively thin and focused on visualization and
interpretation.
"""

from __future__ import annotations

import datetime as dt
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from db.config import SessionLocal, engine
from db.prices import get_price_on_or_before


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_trades_window(
    start_date: dt.date,
    end_date: dt.date,
    chamber: str = "Senate",
) -> pd.DataFrame:
    """Load trades from the DB between ``start_date`` and ``end_date``.

    Parameters
    ----------
    start_date, end_date:
        Date bounds for ``filing_date`` (inclusive).
    chamber:
        Chamber filter (e.g. "Senate" / "House"). Defaults to "Senate".

    Returns
    -------
    pandas.DataFrame
        Raw trades as stored in the ``trades`` table.
    """

    query = text(
        """
        SELECT *
        FROM trades
        WHERE filing_date BETWEEN :start AND :end
          AND chamber = :chamber
        ORDER BY filing_date ASC
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={
                "start": start_date,
                "end": end_date,
                "chamber": chamber,
            },
        )

    return df


# ---------------------------------------------------------------------------
# Trade preparation & basic returns
# ---------------------------------------------------------------------------


def prepare_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and augment raw trades with basic derived columns.

    This function is designed to be the single entrypoint for turning
    raw DB rows into a DataFrame suitable for analysis. It performs:

    * Date parsing for ``transaction_date`` / ``filing_date``.
    * Normalization of ``transaction_type`` into BUY / SELL where
      possible.
    * Creation of a numeric ``direction`` column (+1 for BUY,
      -1 for SELL, 0 otherwise).
    * Creation of a ``trade_quarter`` label (e.g. "2026Q1").
    * Filtering out trades that lack either ``price_at_transaction``
      or ``current_price`` so downstream stats aren't polluted by
      missing pricing.
    """

    if df.empty:
        return df.copy()

    df = df.copy()

    # Ensure dates are proper datetimes
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])

    # Normalize transaction_type where possible; fall back to existing
    # values if they are already canonical BUY / SELL.
    mapping = {
        "Purchase": "BUY",
        "Purchase (Partial)": "BUY",
        "Sale (Full)": "SELL",
        "Sale (Partial)": "SELL",
    }
    df["transaction_type_norm"] = (
        df["transaction_type"].map(mapping).fillna(df["transaction_type"])
    )

    # Direction: BUY -> +1, SELL -> -1, everything else -> 0
    df["direction"] = df["transaction_type_norm"].map({"BUY": 1, "SELL": -1}).fillna(0)

    # Quarter label based on transaction_date
    q = df["transaction_date"].dt.quarter
    y = df["transaction_date"].dt.year
    df["trade_quarter"] = y.astype(str) + "Q" + q.astype(str)

    # Keep only trades with both pricing fields present for return
    # analysis. Rows with missing prices are still in the DB but would
    # distort percentage-return metrics here.
    df = df[df["price_at_transaction"].notna() & df["current_price"].notna()]

    return df


def add_basic_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-trade PnL, percentage return, and annualized return.

    Uses the existing ``price_at_transaction`` and ``current_price``
    columns and the ``direction`` (+1 / -1) to compute:

    * ``shares_est`` – position size implied by ``mid_point``.
    * ``pnl`` – mark-to-market PnL from transaction date to "now".
    * ``pct_return`` – direction-adjusted percentage return.
    * ``days_held`` – days between transaction_date and today.
    * ``ann_return`` – annualized return based on ``pct_return`` and
      ``days_held``.
    """

    if df.empty:
        return df.copy()

    df = df.copy()

    # Estimated size in shares based on the mid-point notional
    df["shares_est"] = df["mid_point"] / df["price_at_transaction"]

    # Mark-to-market PnL (dollars) from transaction_date to "now"
    df["pnl"] = df["direction"] * df["shares_est"] * (
        df["current_price"] - df["price_at_transaction"]
    )

    # Direction-adjusted percent return
    df["pct_return"] = df["direction"] * (
        df["current_price"] / df["price_at_transaction"] - 1.0
    )

    # Holding period and annualized return
    today = dt.date.today()
    days_held = (today - df["transaction_date"].dt.date).apply(lambda d: max(d.days, 1))
    df["days_held"] = days_held
    df["ann_return"] = (1 + df["pct_return"]) ** (365 / df["days_held"]) - 1

    return df


# ---------------------------------------------------------------------------
# Multi-horizon close-to-close returns
# ---------------------------------------------------------------------------


def add_multi_horizon_returns(
    df: pd.DataFrame,
    horizons: Sequence[int] = (1, 7, 30),
) -> pd.DataFrame:
    """Add direction-adjusted close-to-close returns for multiple horizons.

    For each horizon ``h`` in ``horizons`` (in calendar days), this
    computes a column named ``ret_close_close_{h}d`` defined as:

    .. math::

        direction * ( close(t + h) / close(t) - 1 )

    where ``close(t)`` and ``close(t + h)`` are looked up via
    :func:`get_price_on_or_before`, backed by the shared ``PriceCache``
    and yfinance integration.

    Trades for which either start or end prices cannot be found will
    receive ``NaN`` for that horizon.
    """

    if df.empty:
        return df.copy()

    df = df.copy()

    # Only operate on rows with a ticker and transaction_date; others
    # simply receive NaN for all horizons.
    with SessionLocal() as session:
        for h in horizons:
            col = f"ret_close_close_{h}d"
            rets: List[float | None] = []

            for _, row in df.iterrows():
                ticker = row.get("ticker")
                t_date = row.get("transaction_date")
                direction = row.get("direction", 0)

                if pd.isna(ticker) or pd.isna(t_date) or not direction:
                    rets.append(np.nan)
                    continue

                t_date = pd.to_datetime(t_date).date()
                start_price = get_price_on_or_before(session, ticker, t_date)
                end_price = get_price_on_or_before(
                    session, ticker, t_date + dt.timedelta(days=h)
                )

                if start_price is None or end_price is None:
                    rets.append(np.nan)
                else:
                    raw_ret = end_price / start_price - 1.0
                    rets.append(direction * raw_ret)

            df[col] = pd.Series(rets, index=df.index)

    return df


# ---------------------------------------------------------------------------
# Quartile-based stats ("suspiciousness" metrics)
# ---------------------------------------------------------------------------


def add_quartile_flags(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Add top/bottom quartile flags for a given return metric.

    Parameters
    ----------
    df:
        DataFrame with a numeric column named ``metric``.
    metric:
        Column name to base quartiles on, e.g. ``"ret_close_close_7d"``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with two additional boolean columns:
        ``f"{metric}_is_top_quartile"`` and
        ``f"{metric}_is_bottom_quartile"``.
    """

    if df.empty or metric not in df.columns:
        return df.copy()

    df = df.copy()
    vals = df[metric].dropna()
    if vals.empty:
        # Nothing to do
        df[f"{metric}_is_top_quartile"] = False
        df[f"{metric}_is_bottom_quartile"] = False
        return df

    q25, q75 = np.percentile(vals, [25, 75])

    top_flag = f"{metric}_is_top_quartile"
    bottom_flag = f"{metric}_is_bottom_quartile"

    df[top_flag] = df[metric] >= q75
    df[bottom_flag] = df[metric] <= q25

    return df


def quartile_stats_for_senators(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute quartile-based stats for each senator on a given metric.

    For the specified return metric (e.g. ``ret_close_close_7d``), this
    returns one row per ``senator_display_name`` with:

    * ``n_trades`` – number of trades with non-null metric.
    * ``top_share`` – fraction of trades in the top quartile.
    * ``bottom_share`` – fraction of trades in the bottom quartile.
    * ``avg_ret`` – mean of the metric.
    * ``med_ret`` – median of the metric.
    * ``win_rate`` – share of trades with positive metric.
    """

    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    top_flag = f"{metric}_is_top_quartile"
    bottom_flag = f"{metric}_is_bottom_quartile"

    subset = df.dropna(subset=[metric]).copy()
    if subset.empty:
        return pd.DataFrame()

    grouped = (
        subset
        .groupby("senator_display_name")
        .agg(
            n_trades=("id", "count"),
            top_share=(top_flag, "mean"),
            bottom_share=(bottom_flag, "mean"),
            avg_ret=(metric, "mean"),
            med_ret=(metric, "median"),
            win_rate=(metric, lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    return grouped


def quartile_stats_for_tickers(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute quartile-based stats for each ticker on a given metric.

    Same structure as :func:`quartile_stats_for_senators`, but grouped
    by ``ticker`` instead of senator.
    """

    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    top_flag = f"{metric}_is_top_quartile"
    bottom_flag = f"{metric}_is_bottom_quartile"

    subset = df.dropna(subset=[metric]).copy()
    if subset.empty:
        return pd.DataFrame()

    grouped = (
        subset
        .groupby("ticker")
        .agg(
            n_trades=("id", "count"),
            top_share=(top_flag, "mean"),
            bottom_share=(bottom_flag, "mean"),
            avg_ret=(metric, "mean"),
            med_ret=(metric, "median"),
            win_rate=(metric, lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    return grouped


# ---------------------------------------------------------------------------
# Senator-by-quarter summaries
# ---------------------------------------------------------------------------


def top_senators_by_quarter(
    df: pd.DataFrame,
    metric: str,
    min_trades: int = 5,
    top_k: int = 10,
) -> pd.DataFrame:
    """Return top senators by quarter for a given return metric.

    Parameters
    ----------
    df:
        Prepared trades DataFrame with at least ``senator_display_name``,
        ``trade_quarter``, and the specified ``metric`` column.
    metric:
        Return metric to rank by (e.g. ``"pct_return"`` or
        ``"ret_close_close_7d"``).
    min_trades:
        Minimum number of trades in a quarter for a senator to be
        considered.
    top_k:
        Number of senators to keep per quarter.
    """

    if df.empty or metric not in df.columns:
        return pd.DataFrame()

    subset = df.dropna(subset=[metric]).copy()
    if subset.empty:
        return pd.DataFrame()

    grouped = (
        subset
        .groupby(["senator_display_name", "trade_quarter"])
        .agg(
            n_trades=("id", "count"),
            avg_ret=(metric, "mean"),
            med_ret=(metric, "median"),
            win_rate=(metric, lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    grouped = grouped[grouped["n_trades"] >= min_trades]
    if grouped.empty:
        return grouped

    grouped = grouped.sort_values(
        ["trade_quarter", "avg_ret"], ascending=[True, False]
    )

    # Take top_k per quarter
    result = grouped.groupby("trade_quarter").head(top_k).reset_index(drop=True)
    return result


__all__ = [
    "load_trades_window",
    "prepare_trades",
    "add_basic_returns",
    "add_multi_horizon_returns",
    "add_quartile_flags",
    "quartile_stats_for_senators",
    "quartile_stats_for_tickers",
    "top_senators_by_quarter",
]
