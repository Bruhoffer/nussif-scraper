"""Idempotent upsert logic for PTR trades into the local database."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import SessionLocal
from .models import Trade


def upsert_trades(trades: Iterable[Dict[str, Any]]) -> int:
    """Insert new trades, skipping ones that already exist.

    Uses a lookup-before-insert strategy based on the same fields as the
    ``uq_trade_identity`` constraint on ``Trade`` and also deduplicates
    within the current batch to avoid violating the UNIQUE constraint when
    multiple identical trades appear in a single scrape.

    Returns the number of newly inserted rows.
    """

    inserted = 0
    # Track keys we've already processed in this batch to avoid inserting
    # duplicate rows within a single run (in-memory duplicates).
    seen_keys: set[tuple[Any, Any, Any, Any, Any]] = set()

    with SessionLocal() as session:
        for t in trades:
            key = (
                t.get("senator_name"),
                t.get("ticker"),
                t.get("transaction_date"),
                t.get("amount_min"),
                t.get("amount_max"),
            )

            # Skip duplicates within this batch
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Skip if already present in the database
            if _trade_exists(session, t):
                continue

            session.add(_trade_from_dict(t))
            inserted += 1

        session.commit()

    return inserted


def _trade_exists(session: Session, t: Dict[str, Any]) -> bool:
    """Check whether a trade already exists based on identity fields."""

    stmt = select(Trade).where(
        Trade.senator_name == t.get("senator_name"),
        Trade.ticker == t.get("ticker"),
        Trade.transaction_date == t.get("transaction_date"),
        Trade.amount_min == t.get("amount_min"),
        Trade.amount_max == t.get("amount_max"),
    )
    return session.execute(stmt).first() is not None


def _trade_from_dict(t: Dict[str, Any]) -> Trade:
    """Create a Trade ORM instance from a parsed trade dict."""

    return Trade(**t)

