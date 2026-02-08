"""Utilities for enriching ticker metadata (company, sector, industry).

This module finds tickers in the ``trades`` table that lack metadata and
populates the ``ticker_metadata`` table using yfinance.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional

import yfinance as yf
from sqlalchemy import distinct, select
from sqlalchemy.orm import Session

from .config import SessionLocal
from .models import TickerMetadata, Trade



def _find_unknown_tickers(session: Session, limit: Optional[int] = None) -> List[str]:
    """Return tickers present in trades that have no metadata yet.

    A ticker is considered "unknown" if there is no corresponding row in
    ``ticker_metadata``. Null / empty tickers are skipped.
    """

    stmt = (
        select(distinct(Trade.ticker))
        .outerjoin(TickerMetadata, Trade.ticker == TickerMetadata.ticker)
        .where(
            Trade.ticker.is_not(None),
            Trade.ticker != "",
            TickerMetadata.ticker.is_(None),
        )
    )

    rows = session.execute(stmt).scalars().all()
    tickers = [t for t in rows if t]

    if limit is not None:
        return tickers[:limit]
    return tickers


def _fetch_ticker_info_yf(ticker: str) -> dict | None:
    """Fetch basic ticker metadata from yfinance.

    Returns a dict with keys: company_name, sector, industry. If the
    lookup fails or yfinance has no useful data, returns None.
    """

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[ticker_metadata] yfinance lookup failed for {ticker}: {exc}")
        return None

    if not info:
        return None

    company_name = info.get("longName") or info.get("shortName")
    sector = info.get("sector")
    industry = info.get("industry")

    # If we have absolutely nothing useful, skip creating a row so we
    # can try again later (or handle these tickers manually).
    if not any([company_name, sector, industry]):
        return None

    return {
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
    }


def _upsert_ticker_metadata(session: Session, ticker: str, meta: dict) -> None:
    """Insert or update a TickerMetadata row for ``ticker``."""

    obj = session.get(TickerMetadata, ticker)
    today = date.today()

    if obj is None:
        obj = TickerMetadata(
            ticker=ticker,
            company_name=meta.get("company_name"),
            sector=meta.get("sector"),
            industry=meta.get("industry"),
            last_updated=today,
        )
        session.add(obj)
    else:
        obj.company_name = meta.get("company_name", obj.company_name)
        obj.sector = meta.get("sector", obj.sector)
        obj.industry = meta.get("industry", obj.industry)
        obj.last_updated = today


def enrich_ticker_metadata(max_tickers: int = 50) -> int:
    """Enrich missing ticker metadata using yfinance.

    Looks for tickers in ``trades`` that have no entry in
    ``ticker_metadata`` and attempts to fetch company / sector /
    industry data from yfinance. Returns the number of tickers
    successfully updated/inserted.
    """

    updated = 0

    with SessionLocal() as session:
        tickers = _find_unknown_tickers(session, limit=max_tickers)
        if not tickers:
            return 0

        for ticker in tickers:
            meta = _fetch_ticker_info_yf(ticker)
            if meta is None:
                continue

            _upsert_ticker_metadata(session, ticker, meta)
            updated += 1

        session.commit()

    return updated


if __name__ == "__main__":  # simple CLI entrypoint for local use
    count = enrich_ticker_metadata(max_tickers=100)
    print(f"Enriched metadata for {count} tickers.")
