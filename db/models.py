"""SQLAlchemy ORM models for insiderscraper."""

from __future__ import annotations

from sqlalchemy import Column, Date, Float, Integer, String, Text, UniqueConstraint
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


class PortfolioSnapshot(Base):
    """Precomputed monthly portfolio value curve for each senator.

    Populated by ingest/portfolio_snapshots.py during the weekly run.
    The app reads from this table instead of computing curves at runtime.
    """

    __tablename__ = "portfolio_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    senator_display_name = Column(String(200), index=True)
    snapshot_date = Column(Date, index=True)
    portfolio_value = Column(Float)
    last_computed = Column(Date)

    __table_args__ = (
        UniqueConstraint(
            "senator_display_name",
            "snapshot_date",
            name="uq_portfolio_snapshot",
        ),
    )


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


class LobbyingFiling(Base):
    """Corporate lobbying disclosure from the Senate LDA API (lda.senate.gov).

    Each row is one quarterly LD-2 filing. ``client_name`` is the company
    paying for the lobbying; ``registrant_name`` is the lobbying firm hired.
    """

    __tablename__ = "lobbying_filings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filing_uuid = Column(String(100), unique=True, index=True)
    client_name = Column(String(300), index=True)
    ticker = Column(String(32), index=True)       # nullable — mapped by ticker_mapper
    registrant_name = Column(String(300))
    amount = Column(Float)
    filing_date = Column(Date, index=True)
    period_of_lobbying = Column(String(50))        # e.g. "Q1 2026"
    specific_issues = Column(Text)                 # truncated free-text summary
    source_url = Column(String(500))


class GovContract(Base):
    """Federal contract award from USASpending.gov.

    ``award_id`` is the USASpending unique identifier used for upserts.
    """

    __tablename__ = "gov_contracts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    award_id = Column(String(200), unique=True, index=True)
    recipient_name = Column(String(300), index=True)
    ticker = Column(String(32), index=True)        # nullable — mapped by ticker_mapper
    award_amount = Column(Float)
    award_date = Column(Date, index=True)
    funding_agency = Column(String(200))
    description = Column(Text)


class ActivistFiling(Base):
    """SEC Schedule 13D/G activist or passive beneficial-ownership filing.

    ``target_company`` is the issuer (the stock being held).
    ``lead_investor`` is the filer (the activist fund or individual).
    ``accession_number`` is the EDGAR unique identifier used for upserts.
    """

    __tablename__ = "activist_filings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    accession_number = Column(String(30), unique=True, index=True)
    ticker = Column(String(32), index=True)        # nullable — mapped by ticker_mapper
    target_company = Column(String(300), index=True)
    lead_investor = Column(String(300))
    form_type = Column(String(20))                 # SC 13D, SC 13D/A, SC 13G, SC 13G/A
    filing_date = Column(Date, index=True)
    shares = Column(Float)
    prev_shares = Column(Float)
    ownership_pct = Column(Float)
    shares_change_pct = Column(Float)
    market_cap_m = Column(Float)
    sec_url = Column(String(500))

