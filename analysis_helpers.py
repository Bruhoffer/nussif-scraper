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
import yfinance as yf
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


# ---------------------------------------------------------------------------
# Stateful position tracker
# ---------------------------------------------------------------------------


def _normalise_senator_df(senator_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to lowercase internal names.

    The app layer renames DB columns to Title Case for display.  All
    analysis functions work with the lowercase internal names so they
    are independent of the UI renaming.
    """
    df = senator_df.copy()
    rename = {
        "Transaction Date": "transaction_date",
        "Filing Date": "filing_date",
        "Senator": "senator_display_name",
        "Type": "transaction_type",
        "Amount Range": "amount_range_raw",
        "Mid Point": "mid_point",
        "Chamber": "chamber",
        "Ticker": "ticker",
        "Sector": "sector",
        "Price At Transaction": "price_at_transaction",
        "Current Price": "current_price",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    # Drop duplicate columns produced when both Title Case and lowercase versions
    # exist (e.g. all_trades_df carries both price_at_transaction and
    # Price At Transaction). Keep the first occurrence.
    df = df.loc[:, ~df.columns.duplicated()]

    _empty_float = pd.Series(dtype=float)
    _empty_str   = pd.Series(dtype=str)

    df["transaction_date"]    = pd.to_datetime(df["transaction_date"] if "transaction_date" in df.columns else _empty_str, errors="coerce")
    df["mid_point"]           = pd.to_numeric(df["mid_point"]           if "mid_point"           in df.columns else _empty_float, errors="coerce").fillna(0)
    df["price_at_transaction"]= pd.to_numeric(df["price_at_transaction"] if "price_at_transaction" in df.columns else _empty_float, errors="coerce")
    df["current_price"]       = pd.to_numeric(df["current_price"]       if "current_price"       in df.columns else _empty_float, errors="coerce")
    df["transaction_type"]    = (df["transaction_type"] if "transaction_type" in df.columns else _empty_str).str.upper()
    if "sector" not in df.columns:
        df["sector"] = "Unknown"

    return df.sort_values("transaction_date").reset_index(drop=True)


def track_positions(senator_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk trades chronologically and track position state per ticker.

    Position state machine (per ticker):
        flat (0)  + BUY  → open long  (shares > 0)
        long      + BUY  → scale into long
        long      + SELL → close long, realise P&L
        flat (0)  + SELL → open short (shares < 0)
        short     + SELL → scale into short
        short     + BUY  → close short, realise P&L

    Shares are estimated from mid_point / price_at_transaction because
    actual share counts are not disclosed — values are directionally
    correct but not precise.

    Parameters
    ----------
    senator_df:
        Full trade history for one senator (all time, all tickers).
        Accepts both Title Case (app layer) and lowercase (DB layer) columns.

    Returns
    -------
    open_positions : pd.DataFrame
        One row per currently open position (long or short) with columns:
        ticker, sector, direction, shares, avg_entry_price, current_price,
        cost_basis, current_value, unrealized_pnl, roi_pct, opened_date,
        last_trade_date.

    closed_trades : pd.DataFrame
        One row per completed round-trip with columns:
        ticker, direction (of the closed leg), shares, entry_price,
        exit_price, cost_basis, proceeds, realised_pnl, roi_pct,
        open_date, close_date, holding_days.
    """

    if senator_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = _normalise_senator_df(senator_df)
    df = df[df["ticker"].notna() & (df["ticker"] != "")]

    # positions[ticker] = {"shares": float, "avg_entry_price": float,
    #                       "cost_basis": float, "opened_date": date,
    #                       "sector": str}
    positions: dict[str, dict] = {}
    closed_rows: list[dict] = []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        tx_type = row["transaction_type"]
        price = row["price_at_transaction"]
        mid = row["mid_point"]
        tx_date = row["transaction_date"]
        sector = row.get("sector") or "Unknown"

        if pd.isna(price) or price <= 0 or mid <= 0:
            continue
        if tx_type not in ("BUY", "SELL"):
            continue

        shares_transacted = mid / price
        pos = positions.get(ticker)

        if tx_type == "BUY":
            if pos is None or pos["shares"] == 0:
                # Open new long (or re-enter after flat)
                positions[ticker] = {
                    "shares": shares_transacted,
                    "avg_entry_price": price,
                    "cost_basis": mid,
                    "opened_date": tx_date,
                    "sector": sector,
                }
            elif pos["shares"] > 0:
                # Scale into existing long — update avg entry price
                total_shares = pos["shares"] + shares_transacted
                total_cost = pos["cost_basis"] + mid
                positions[ticker] = {
                    "shares": total_shares,
                    "avg_entry_price": total_cost / total_shares,
                    "cost_basis": total_cost,
                    "opened_date": pos["opened_date"],
                    "sector": sector,
                }
            else:
                # pos["shares"] < 0 → closing a short position
                closed_shares = min(shares_transacted, abs(pos["shares"]))
                proceeds = closed_shares * pos["avg_entry_price"]  # what we received when shorting
                cost_to_close = closed_shares * price
                realised_pnl = proceeds - cost_to_close
                roi_pct = (realised_pnl / proceeds * 100) if proceeds else None

                closed_rows.append({
                    "Ticker": ticker,
                    "Direction": "SHORT",
                    "Shares": closed_shares,
                    "Entry Price": pos["avg_entry_price"],
                    "Exit Price": price,
                    "Cost Basis": proceeds,
                    "Proceeds": cost_to_close,
                    "Realised P&L": realised_pnl,
                    "ROI (%)": roi_pct,
                    "Open Date": pos["opened_date"],
                    "Close Date": tx_date,
                    "Holding Days": max((tx_date - pos["opened_date"]).days, 0) if pd.notna(pos["opened_date"]) else None,
                })

                remaining = pos["shares"] + shares_transacted  # shares < 0, adding positive
                if remaining >= 0:
                    positions[ticker] = {"shares": 0, "avg_entry_price": 0, "cost_basis": 0,
                                         "opened_date": None, "sector": sector}
                else:
                    positions[ticker] = {**pos, "shares": remaining,
                                         "cost_basis": abs(remaining) * pos["avg_entry_price"]}

        else:  # SELL
            if pos is None or pos["shares"] == 0:
                # Open new short
                positions[ticker] = {
                    "shares": -shares_transacted,
                    "avg_entry_price": price,
                    "cost_basis": mid,
                    "opened_date": tx_date,
                    "sector": sector,
                }
            elif pos["shares"] < 0:
                # Scale into existing short
                total_shares = pos["shares"] - shares_transacted  # more negative
                total_cost = pos["cost_basis"] + mid
                positions[ticker] = {
                    "shares": total_shares,
                    "avg_entry_price": total_cost / abs(total_shares),
                    "cost_basis": total_cost,
                    "opened_date": pos["opened_date"],
                    "sector": sector,
                }
            else:
                # pos["shares"] > 0 → closing a long position
                closed_shares = min(shares_transacted, pos["shares"])
                cost_basis = closed_shares * pos["avg_entry_price"]
                proceeds = closed_shares * price
                realised_pnl = proceeds - cost_basis
                roi_pct = (realised_pnl / cost_basis * 100) if cost_basis else None

                closed_rows.append({
                    "Ticker": ticker,
                    "Direction": "LONG",
                    "Shares": closed_shares,
                    "Entry Price": pos["avg_entry_price"],
                    "Exit Price": price,
                    "Cost Basis": cost_basis,
                    "Proceeds": proceeds,
                    "Realised P&L": realised_pnl,
                    "ROI (%)": roi_pct,
                    "Open Date": pos["opened_date"],
                    "Close Date": tx_date,
                    "Holding Days": max((tx_date - pos["opened_date"]).days, 0) if pd.notna(pos["opened_date"]) else None,
                })

                remaining = pos["shares"] - shares_transacted
                if remaining <= 0:
                    positions[ticker] = {"shares": 0, "avg_entry_price": 0, "cost_basis": 0,
                                         "opened_date": None, "sector": sector}
                else:
                    positions[ticker] = {**pos, "shares": remaining,
                                         "cost_basis": remaining * pos["avg_entry_price"]}

    # Build open positions DataFrame
    open_rows: list[dict] = []
    for ticker, pos in positions.items():
        if pos["shares"] == 0:
            continue

        current_price = df[df["ticker"] == ticker]["current_price"].dropna()
        cp = float(current_price.iloc[-1]) if not current_price.empty else None

        shares = pos["shares"]
        avg_entry = pos["avg_entry_price"]
        cost_basis = pos["cost_basis"]
        direction = "LONG" if shares > 0 else "SHORT"

        if cp is not None:
            current_value = abs(shares) * cp
            if direction == "LONG":
                unrealized_pnl = current_value - cost_basis
            else:
                unrealized_pnl = cost_basis - current_value  # short: profit when price falls
            roi_pct = (unrealized_pnl / cost_basis * 100) if cost_basis else None
        else:
            current_value = None
            unrealized_pnl = None
            roi_pct = None

        sector = df[df["ticker"] == ticker]["sector"].dropna()
        sector_val = sector.iloc[0] if not sector.empty else "Unknown"

        open_rows.append({
            "Ticker": ticker,
            "Sector": sector_val,
            "Direction": direction,
            "Shares (Est)": abs(shares),
            "Avg Entry Price": avg_entry,
            "Current Price": cp,
            "Cost Basis": cost_basis,
            "Current Value": current_value,
            "Unrealized P&L": unrealized_pnl,
            "ROI (%)": roi_pct,
            "Opened Date": pos["opened_date"],
            "Last Trade Date": df[df["ticker"] == ticker]["transaction_date"].max(),
        })

    open_positions = pd.DataFrame(open_rows)
    if not open_positions.empty:
        open_positions = open_positions.sort_values("Current Value", ascending=False, na_position="last").reset_index(drop=True)

    closed_trades = pd.DataFrame(closed_rows) if closed_rows else pd.DataFrame()

    return open_positions, closed_trades


# ---------------------------------------------------------------------------
# Portfolio curve
# ---------------------------------------------------------------------------


def compute_portfolio_curve(
    senator_df: pd.DataFrame,
    freq: str = "ME",
    start_from: dt.date | None = None,
) -> pd.DataFrame:
    """Build a time series of estimated portfolio value.

    For each period-end date from the senator's first trade to today,
    sums the mark-to-market value of all open positions at that date
    using the shared PriceCache (backed by yfinance).

    Short positions contribute negative value (mark-to-market loss if
    price rises, gain if price falls).

    Parameters
    ----------
    senator_df:
        Full trade history for one senator.
    freq:
        Pandas offset alias for the resampling frequency.
        "ME" = month-end (default), "W" = weekly, "QE" = quarter-end.
    start_from:
        If provided, only compute period-ends strictly after this date.
        Used for delta updates — pass the last stored snapshot date so
        only new months are computed instead of the full history.
        Note: even when start_from is set, the full trade history in
        senator_df is still used to correctly replay positions at each
        new period-end (historical trades affect current open positions).

    Returns
    -------
    pd.DataFrame with columns: date, portfolio_value.
    Empty DataFrame if there is insufficient data.
    """

    if senator_df.empty:
        return pd.DataFrame()

    df = _normalise_senator_df(senator_df)
    df = df[df["ticker"].notna() & (df["ticker"] != "")]

    first_date = df["transaction_date"].min()
    if pd.isna(first_date):
        return pd.DataFrame()

    today = dt.date.today()
    period_ends = pd.date_range(start=first_date, end=today, freq=freq).date.tolist()
    if not period_ends or period_ends[-1] < today:
        period_ends.append(today)

    # Delta mode: skip period-ends already stored in the DB
    if start_from is not None:
        period_ends = [d for d in period_ends if d > start_from]
        if not period_ends:
            return pd.DataFrame()

    # --- Single-pass position tracking ---
    # Walk trades and period-ends together in one pass instead of replaying
    # track_positions() from scratch at every period-end (O(n) vs O(n²)).
    #
    # positions[ticker] mirrors the state dict used in track_positions():
    #   {"shares": float, "avg_entry_price": float, "cost_basis": float}
    positions: dict[str, dict] = {}
    trade_idx = 0
    trades_list = df.to_dict("records")  # avoid repeated DataFrame slicing
    n_trades = len(trades_list)

    # --- Pre-fetch all required prices into PriceCache ---
    # Collect every (ticker, period_date) pair we will need, then fetch them
    # all up front so the main loop only does fast DB reads, not yfinance calls.
    # Track failed tickers so each bad symbol is only attempted once, not once
    # per period-end (avoids dozens of identical yfinance error messages).
    all_tickers = df["ticker"].dropna().unique().tolist()
    failed_tickers: set[str] = set()
    with SessionLocal() as session:
        for ticker in all_tickers:
            if ticker in failed_tickers:
                continue
            for period_date in period_ends:
                price = get_price_on_or_before(session, ticker, period_date)
                if price is None:
                    # First miss — mark as failed so we skip remaining dates
                    failed_tickers.add(ticker)
                    break
        session.commit()

    # --- Main loop: single pass over period_ends ---
    curve_rows: list[dict] = []

    with SessionLocal() as session:
        for period_date in period_ends:
            # Advance trades up to and including this period_date
            while trade_idx < n_trades:
                row = trades_list[trade_idx]
                tx_date = row["transaction_date"]
                if isinstance(tx_date, pd.Timestamp):
                    tx_date = tx_date.date()
                if tx_date > period_date:
                    break

                ticker = row.get("ticker")
                tx_type = str(row.get("transaction_type") or "").upper()
                price = row.get("price_at_transaction")
                mid = row.get("mid_point") or 0

                if ticker and tx_type in ("BUY", "SELL") and price and price > 0 and mid > 0:
                    shares_transacted = mid / price
                    pos = positions.get(ticker, {"shares": 0, "avg_entry_price": 0, "cost_basis": 0})

                    if tx_type == "BUY":
                        if pos["shares"] >= 0:
                            total_shares = pos["shares"] + shares_transacted
                            total_cost = pos["cost_basis"] + mid
                            positions[ticker] = {
                                "shares": total_shares,
                                "avg_entry_price": total_cost / total_shares,
                                "cost_basis": total_cost,
                            }
                        else:  # closing short
                            remaining = pos["shares"] + shares_transacted
                            positions[ticker] = {
                                "shares": remaining,
                                "avg_entry_price": pos["avg_entry_price"],
                                "cost_basis": abs(remaining) * pos["avg_entry_price"] if remaining < 0 else 0,
                            }
                    else:  # SELL
                        if pos["shares"] <= 0:
                            # No confirmed prior long — could be RSUs, pre-data position,
                            # or ambiguous. Skip rather than opening a phantom short that
                            # would drive the portfolio value negative.
                            pass
                        else:  # closing long
                            remaining = pos["shares"] - shares_transacted
                            positions[ticker] = {
                                "shares": remaining,
                                "avg_entry_price": pos["avg_entry_price"],
                                "cost_basis": remaining * pos["avg_entry_price"] if remaining > 0 else 0,
                            }

                trade_idx += 1

            # Mark open positions to market at this period_date
            total_value = 0.0
            for ticker, pos in positions.items():
                if pos["shares"] == 0 or ticker in failed_tickers:
                    continue
                price = get_price_on_or_before(session, ticker, period_date)
                if price is None:
                    continue
                pos_value = abs(pos["shares"]) * price
                total_value += pos_value if pos["shares"] > 0 else -pos_value

            curve_rows.append({"date": period_date, "portfolio_value": total_value})

    return pd.DataFrame(curve_rows)


# ---------------------------------------------------------------------------
# Portfolio metrics: max drawdown, beta, sharpe, win rate
# ---------------------------------------------------------------------------


def compute_portfolio_metrics(
    curve_df: pd.DataFrame,
    closed_trades: pd.DataFrame,
) -> dict:
    """Compute summary risk/return metrics from the portfolio curve.

    Parameters
    ----------
    curve_df:
        Output of compute_portfolio_curve — columns: date, portfolio_value.
    closed_trades:
        Output of track_positions — completed round-trips with Realised P&L.

    Returns
    -------
    dict with keys:
        max_drawdown_pct   – worst peak-to-trough drawdown (negative %)
        beta               – portfolio beta vs SPY (None if insufficient data)
        sharpe             – annualised Sharpe ratio (None if insufficient data)
        win_rate_pct       – % of closed trades with positive realised P&L
        total_realised_pnl – sum of all closed trade P&L
    """

    result = {
        "max_drawdown_pct": None,
        "beta": None,
        "sharpe": None,
        "win_rate_pct": None,
        "total_realised_pnl": None,
    }

    # --- Win rate & realised P&L from closed trades ---
    if not closed_trades.empty and "Realised P&L" in closed_trades.columns:
        pnl = closed_trades["Realised P&L"].dropna()
        if not pnl.empty:
            result["win_rate_pct"] = float((pnl > 0).mean() * 100)
            result["total_realised_pnl"] = float(pnl.sum())

    if curve_df.empty or len(curve_df) < 3:
        return result

    values = curve_df["portfolio_value"].values.astype(float)

    # --- Max drawdown ---
    peak = np.maximum.accumulate(values)
    # Only compute where peak > 0; if peak never goes positive (e.g. all-short
    # portfolio or insufficient data), drawdown is undefined — leave as None.
    valid = peak > 0
    if valid.any():
        drawdowns = np.where(valid, (values - peak) / peak, 0.0)
        result["max_drawdown_pct"] = float(drawdowns.min() * 100)
    else:
        result["max_drawdown_pct"] = None

    # --- Monthly returns for Sharpe and Beta ---
    monthly = (
        curve_df
        .assign(date=pd.to_datetime(curve_df["date"]))
        .set_index("date")["portfolio_value"]
        .resample("ME").last()
        .dropna()
    )
    if len(monthly) < 3:
        return result

    port_returns = monthly.pct_change().dropna()
    if port_returns.empty:
        return result

    # Sharpe (annualised, risk-free ≈ 0 for simplicity)
    if port_returns.std() > 0:
        result["sharpe"] = float((port_returns.mean() / port_returns.std()) * np.sqrt(12))

    # Beta vs SPY
    try:
        spy_hist = yf.download("SPY", start=monthly.index[0], end=monthly.index[-1], progress=False)
        if not spy_hist.empty:
            spy_monthly = spy_hist["Close"].resample("ME").last().pct_change().dropna()
            aligned = port_returns.align(spy_monthly, join="inner")
            p_ret, m_ret = aligned[0].values, aligned[1].values
            if len(p_ret) >= 3 and m_ret.std() > 0:
                cov = np.cov(p_ret, m_ret)
                result["beta"] = float(cov[0, 1] / cov[1, 1])
    except Exception:
        pass  # Beta is best-effort; missing SPY data should not break the page

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
    "track_positions",
    "compute_portfolio_curve",
    "compute_portfolio_metrics",
]
