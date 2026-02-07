"""SQLAlchemy ORM models for insiderscraper."""

from __future__ import annotations

from sqlalchemy import Column, Date, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Trade(Base):
    """Single parsed PTR trade.

    Columns mirror the dicts produced by ``scraper.ptr_details`` so that we
    can simply unpack trade dicts into this model for inserts.
    """

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Use bounded String lengths so that Azure SQL can index / enforce
    # unique constraints on these columns (SQL Server does not allow
    # VARCHAR(MAX) in index/unique key definitions).

    senator_name = Column(String(200), index=True)
    senator_first_name = Column(String(100))
    senator_last_name = Column(String(100))
    senator_display_name = Column(String(200))
    chamber = Column(String(20))

    report_id = Column(String(100), index=True)
    report_type = Column(String(50))
    report_format = Column(String(20))
    filing_date = Column(Date, index=True)

    transaction_date = Column(Date, index=True)
    owner = Column(String(50))
    ticker = Column(String(32), index=True)
    asset_name = Column(String(300))
    asset_type = Column(String(100))

    transaction_type = Column(String(20))
    transaction_type_raw = Column(String(100))

    amount_range_raw = Column(String(100))
    amount_min = Column(Float)
    amount_max = Column(Float)
    mid_point = Column(Float)

    comment = Column(String(500))

    __table_args__ = (
        UniqueConstraint(
            "senator_name",
            "ticker",
            "transaction_date",
            "amount_min",
            "amount_max",
            name="uq_trade_identity",
        ),
    )

