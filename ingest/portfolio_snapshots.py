"""Precompute and store monthly portfolio curves for all senators.

Run after the weekly scrape so prices are already in the PriceCache.
Idempotent — existing snapshots are updated in place.

Usage:
    python -m ingest.portfolio_snapshots
    python -m ingest.portfolio_snapshots --senator "Nancy Pelosi"  # single senator
"""

from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd
from sqlalchemy import select, text, func

from db.config import SessionLocal, init_db, engine
from db.models import PortfolioSnapshot


def _get_all_senators() -> list[str]:
    """Return all distinct senator_display_name values in the trades table."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT senator_display_name FROM trades WHERE senator_display_name IS NOT NULL")
        ).fetchall()
    return [r[0] for r in rows if r[0]]


def _load_senator_trades(senator: str) -> pd.DataFrame:
    """Load full trade history for one senator with all columns needed by track_positions."""
    query = text(
        """
        SELECT t.transaction_date, t.transaction_type, t.mid_point,
               t.price_at_transaction, t.current_price, t.ticker,
               m.sector
        FROM trades t
        LEFT JOIN ticker_metadata m ON t.ticker = m.ticker
        WHERE t.senator_display_name = :senator
          AND t.transaction_date IS NOT NULL
        ORDER BY t.transaction_date ASC
        """
    )
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"senator": senator})


def _get_last_snapshot_date(senator: str) -> dt.date | None:
    """Return the most recent snapshot_date stored for this senator, or None."""
    with SessionLocal() as session:
        result = session.execute(
            select(func.max(PortfolioSnapshot.snapshot_date)).where(
                PortfolioSnapshot.senator_display_name == senator
            )
        ).scalar()
    return result  # None if no rows exist yet


def _upsert_snapshots(senator: str, curve_df: pd.DataFrame) -> int:
    """Write portfolio curve rows into portfolio_snapshots, upserting by (senator, date)."""
    if curve_df.empty:
        return 0

    today = dt.date.today()
    upserted = 0

    with SessionLocal() as session:
        for _, row in curve_df.iterrows():
            snap_date = row["date"]
            if isinstance(snap_date, pd.Timestamp):
                snap_date = snap_date.date()

            existing = session.execute(
                select(PortfolioSnapshot).where(
                    PortfolioSnapshot.senator_display_name == senator,
                    PortfolioSnapshot.snapshot_date == snap_date,
                )
            ).scalars().first()

            if existing is None:
                session.add(PortfolioSnapshot(
                    senator_display_name=senator,
                    snapshot_date=snap_date,
                    portfolio_value=float(row["portfolio_value"]),
                    last_computed=today,
                ))
            else:
                existing.portfolio_value = float(row["portfolio_value"])
                existing.last_computed = today

            upserted += 1

        session.commit()

    return upserted


def _process_senator(senator: str) -> tuple[str, int, str]:
    """Load trades, compute curve, and upsert snapshots for one senator.

    Returns (senator, rows_upserted, status_message).
    Designed to run inside a thread-pool worker.
    """
    from analysis_helpers import compute_portfolio_curve

    trades_df = _load_senator_trades(senator)
    if trades_df.empty:
        return senator, 0, "no trades"

    last_date = _get_last_snapshot_date(senator)
    curve_df = compute_portfolio_curve(trades_df, start_from=last_date)

    if curve_df.empty:
        return senator, 0, "curve empty"

    count = _upsert_snapshots(senator, curve_df)
    delta_label = f"delta from {last_date}" if last_date else "full history"
    return senator, count, f"upserted {count} rows ({delta_label})"


def run_snapshot_ingest(senator_filter: str | None = None) -> None:
    """Compute and store portfolio curves for all (or one) senator(s).

    Runs sequentially — parallelism is counter-productive here because the
    bottleneck is yfinance rate limits and SQLite price_cache writes, both
    of which serialize under concurrent load.

    Parameters
    ----------
    senator_filter:
        If provided, only compute for this senator. Otherwise processes all.
    """
    init_db()

    senators = [senator_filter] if senator_filter else _get_all_senators()
    total = len(senators)
    print(f"[portfolio_snapshots] Computing curves for {total} senator(s)...")

    for i, senator in enumerate(senators):
        senator, count, msg = _process_senator(senator)
        print(f"[portfolio_snapshots] [{i + 1}/{total}] {senator} → {msg}")

    print("[portfolio_snapshots] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute portfolio curves for all senators")
    parser.add_argument(
        "--senator",
        type=str,
        default=None,
        help="Only compute for this senator (exact display name). Default: all senators.",
    )
    args = parser.parse_args()
    run_snapshot_ingest(senator_filter=args.senator)


if __name__ == "__main__":
    main()
