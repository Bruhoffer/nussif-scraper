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

    # Enriched pricing fields
    #
    # price_at_transaction: closing price on or before the transaction_date
    # current_price: latest known market price for the ticker at ingest time
    #
    # These are kept nullable so existing rows remain valid and we can
    # gradually backfill historical data.
    price_at_transaction = Column(Float)
    current_price = Column(Float)

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


class TickerMetadata(Base):
    """Metadata for traded tickers (company, sector, industry).

    This is populated via external data sources (e.g. yfinance) and
    joined to ``trades`` for sector/industry analytics in the dashboard.
    """

    __tablename__ = "ticker_metadata"

    ticker = Column(String(32), primary_key=True)
    company_name = Column(String(300))
    sector = Column(String(100), index=True)
    industry = Column(String(200), index=True)
    last_updated = Column(Date)


class PriceCache(Base):
    """Cached daily prices for tickers.

    Used as a local cache in front of yfinance so that both the local
    ingest script and Azure Functions stay within rate limits and we
    avoid redundant API calls.
    """

    __tablename__ = "price_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(32), index=True)
    date = Column(Date, index=True)
    price = Column(Float)
    last_updated = Column(Date)

    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_price_cache_ticker_date"),
    )


