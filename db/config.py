"""Database configuration for insiderscraper.

Defaults to a local SQLite database under ``data/ptr_trades.db`` but can be
overridden via the ``DATABASE_URL`` environment variable.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/ptr_trades.db")

engine = create_engine(DATABASE_URL, echo=False, future=True)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create database tables if they do not already exist."""

    from .models import Base  # local import to avoid circular dependency

    Base.metadata.create_all(bind=engine)

