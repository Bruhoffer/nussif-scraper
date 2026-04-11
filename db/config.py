"""Database configuration for insiderscraper.

Defaults to a local SQLite database under ``data/ptr_trades.db`` but can be
overridden via the ``DATABASE_URL`` environment variable.
"""

from __future__ import annotations

import os

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/ptr_trades.db")

_sqlite_args = {}
if DATABASE_URL.startswith("sqlite"):
    # timeout: seconds a thread waits for the write lock before raising.
    # check_same_thread=False: required when the engine is shared across threads.
    _sqlite_args = {"timeout": 60, "check_same_thread": False}

engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=_sqlite_args)


@sqlalchemy.event.listens_for(engine, "connect")
def _set_wal_mode(dbapi_conn, _connection_record):
    """Enable WAL journal mode for better concurrent read/write performance."""
    if DATABASE_URL.startswith("sqlite"):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create database tables if they do not already exist, then run
    idempotent data migrations (e.g. normalise senator display names)."""

    import re
    from sqlalchemy import text
    from .models import Base  # local import to avoid circular dependency

    Base.metadata.create_all(bind=engine)

    # Normalise any remaining "Last, First (Senator)" display names to "First Last".
    # Safe to re-run: rows already in canonical form are not matched by the LIKE filter.
    def _normalise(role: str) -> str:
        cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", role).strip()
        if "," not in cleaned:
            return cleaned
        last, first = cleaned.split(",", 1)
        return f"{first.strip()} {last.strip()}"

    with engine.connect() as conn:
        old_names = conn.execute(text(
            "SELECT DISTINCT senator_display_name FROM trades "
            "WHERE senator_display_name LIKE '%(Senator)%' "
            "   OR senator_display_name LIKE '%(Representative)%'"
        )).fetchall()
        for (old,) in old_names:
            new = _normalise(old)
            if new != old:
                conn.execute(
                    text("UPDATE trades SET senator_display_name = :new WHERE senator_display_name = :old"),
                    {"new": new, "old": old},
                )
        conn.commit()

