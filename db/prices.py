"""Price cache utilities backed by yfinance.

This module provides helpers to look up historical and latest prices
for tickers while caching results in the ``price_cache`` table to avoid
redundant API calls.

Phase B responsibilities (from the high-level plan):
* Fetch ``price_at_transaction``: close on or before ``transaction_date``.
* Fetch ``current_price``: latest close for each ticker.
* Store all fetched prices in ``price_cache`` for reuse.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Optional

import yfinance as yf
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from .config import SessionLocal
from .models import PriceCache


def get_price_on_or_before(
    session: Session,
    ticker: str,
    target_date: dt.date,
) -> Optional[float]:
    """Return close price on or before ``target_date`` for ``ticker``.

    Logic:
    1. Look in ``price_cache`` for this ticker with ``date <= target_date``,
       ordered by date descending; if found, return immediately.
    2. If not found, download ~30 days of history around ``target_date``
       from yfinance, insert those closes into the cache, and re-query.

    This handles weekends/holidays naturally because we always choose
    the closest prior trading day.
    """

    # 1. Check cache first.
    stmt = (
        select(PriceCache)
        .where(
            PriceCache.ticker == ticker,
            PriceCache.date <= target_date,
        )
        .order_by(PriceCache.date.desc())
        .limit(1)
    )
    cached = session.execute(stmt).scalars().first()
    if cached is not None:
        return cached.price

    # 2. Cache miss: fetch a small window of history from yfinance.
    start = target_date - dt.timedelta(days=30)
    end = target_date + dt.timedelta(days=1)

    try:
        hist = yf.download(ticker, start=start, end=end, progress=False)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[prices] yfinance download failed for {ticker}: {exc}")
        return None

    if hist.empty:
        return None

    # Insert all returned daily prices into the cache.
    for idx, row in hist.iterrows():
        day = idx.date()
        close_val = row["Close"]
        # If this is a 1-element Series, use iloc[0]; otherwise just cast
        if hasattr(close_val, "iloc"):
            price = float(close_val.iloc[0])
        else:
            price = float(close_val)

        existing = session.execute(
            select(PriceCache).where(
                and_(PriceCache.ticker == ticker, PriceCache.date == day)
            )
        ).scalars().first()

        if existing is None:
            session.add(
                PriceCache(
                    ticker=ticker,
                    date=day,
                    price=price,
                    last_updated=target_date,
                )
            )

    session.flush()

    # 3. Re-query cache for the desired date (or closest prior).
    cached = session.execute(stmt).scalars().first()
    return cached.price if cached is not None else None


def get_latest_price(session: Session, ticker: str) -> Optional[float]:
    """Return the most recent available close for ``ticker``.

    We download recent history from yfinance (~30 days) and update the
    cache with the last available close.
    """

    today = dt.date.today()
    start = today - dt.timedelta(days=30)

    try:
        hist = yf.download(
            ticker,
            start=start,
            end=today + dt.timedelta(days=1),
            progress=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[prices] yfinance latest price download failed for {ticker}: {exc}")
        return None

    if hist.empty:
        return None

    last_idx = hist.index[-1]
    last_date = last_idx.date()
    # Access the last close value via the Close series so we get a
    # scalar instead of a single-element Series, avoiding pandas'
    # FutureWarning about float(Series).
    last_close = hist["Close"].iloc[-1]
    # If last_close is a Series (e.g. from a MultiIndex/DataFrame), take
    # the first element explicitly; otherwise just cast.
    if hasattr(last_close, "iloc"):
        last_price = float(last_close.iloc[0])
    else:
        last_price = float(last_close)

    existing = session.execute(
        select(PriceCache).where(
            and_(PriceCache.ticker == ticker, PriceCache.date == last_date)
        )
    ).scalars().first()

    if existing is None:
        session.add(
            PriceCache(
                ticker=ticker,
                date=last_date,
                price=last_price,
                last_updated=today,
            )
        )
        session.flush()
    else:
        existing.price = last_price
        existing.last_updated = today

    return last_price


def enrich_prices_for_trades(df) -> None:
    """Mutate a trades DataFrame in-place with price columns.

    Adds ``price_at_transaction`` and ``current_price`` using the
    shared ``PriceCache`` table and yfinance as a backing source.

    This is designed to be called from the dashboard layer (or
    ingestion scripts) after trades are loaded from the DB.
    """

    if df.empty or "ticker" not in df.columns or "transaction_date" not in df.columns:
        return

    with SessionLocal() as session:
        # Map (ticker, transaction_date) -> price_at_transaction
        df_nonnull = df.dropna(subset=["ticker", "transaction_date"]).copy()
        distinct_pairs = (
            df_nonnull[["ticker", "transaction_date"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        price_at_tx: Dict[tuple, float] = {}
        for ticker, tx_date in distinct_pairs:
            if not ticker:
                continue
            price = get_price_on_or_before(session, ticker, tx_date)
            if price is not None:
                price_at_tx[(ticker, tx_date)] = price

        df["price_at_transaction"] = [
            price_at_tx.get((t, d)) if (t is not None and d is not None) else None
            for t, d in zip(df.get("ticker"), df.get("transaction_date"))
        ]

        # Map ticker -> latest price
        tickers = sorted({t for t in df.get("ticker").dropna().unique() if t})
        latest_prices: Dict[str, float] = {}
        for ticker in tickers:
            price = get_latest_price(session, ticker)
            if price is not None:
                latest_prices[ticker] = price

        df["current_price"] = [
            latest_prices.get(t) if t is not None else None
            for t in df.get("ticker")
        ]

        # Persist any new/updated cache entries so subsequent runs (and
        # the Streamlit app) can rely on DB-only reads when needed.
        session.commit()


