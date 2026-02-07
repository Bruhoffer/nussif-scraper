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

    senator_name = Column(String, index=True)
    senator_first_name = Column(String)
    senator_last_name = Column(String)
    senator_display_name = Column(String)
    chamber = Column(String)

    report_id = Column(String, index=True)
    report_type = Column(String)
    report_format = Column(String)
    filing_date = Column(Date, index=True)

    transaction_date = Column(Date, index=True)
    owner = Column(String)
    ticker = Column(String, index=True)
    asset_name = Column(String)
    asset_type = Column(String)

    transaction_type = Column(String)
    transaction_type_raw = Column(String)

    amount_range_raw = Column(String)
    amount_min = Column(Float)
    amount_max = Column(Float)
    mid_point = Column(Float)

    comment = Column(String)

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

