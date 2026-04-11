"""Map company names to stock tickers using the local ticker_metadata table.

Used by all three Market Intelligence ingest pipelines (lobbying, gov_contracts,
activist_filings) to attach a ticker symbol to each company name returned by
external APIs, which only provide free-text company names.

Strategy:
1. Normalise both sides (lowercase, strip punctuation / legal suffixes).
2. Exact match against ticker_metadata.company_name.
3. Fuzzy match (RapidFuzz token_sort_ratio) with threshold 85.
4. Return None if no match found — rows still stored, just without a ticker.

The mapper loads ticker_metadata once per process and caches results in memory
so repeated lookups for the same company name don't re-query the DB.
"""

from __future__ import annotations

import re
from functools import lru_cache

from sqlalchemy import text

from db.config import engine

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES = re.compile(
    r"\b(inc|corp|co|ltd|llc|lp|plc|group|holdings|international|"
    r"technologies|technology|solutions|services|enterprises|partners|"
    r"associates|trust|fund|capital|management|mgmt|sa|ag|nv|bv|"
    r"class\s+[a-z]|ordinary shares?|depositary receipt)\b",
    re.IGNORECASE,
)
_NON_ALPHA = re.compile(r"[^a-z0-9\s]")


def _normalise(name: str) -> str:
    """Lowercase, strip legal suffixes and punctuation for comparison."""
    s = name.lower()
    s = _LEGAL_SUFFIXES.sub("", s)
    s = _NON_ALPHA.sub(" ", s)
    return " ".join(s.split())   # collapse whitespace


# ---------------------------------------------------------------------------
# Metadata loader (loaded once, cached for the process lifetime)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_metadata() -> dict[str, str]:
    """Return {normalised_company_name: ticker} from ticker_metadata."""
    query = text("SELECT ticker, company_name FROM ticker_metadata WHERE company_name IS NOT NULL")
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    return {_normalise(r[1]): r[0] for r in rows if r[1]}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Per-run cache: {raw_company_name: ticker_or_None}
_session_cache: dict[str, str | None] = {}


def map_company_to_ticker(company_name: str, threshold: int = 85) -> str | None:
    """Return the best-matching ticker for ``company_name``, or None.

    Parameters
    ----------
    company_name:
        Raw company name string from an external API.
    threshold:
        Minimum RapidFuzz token_sort_ratio score (0–100) to accept a match.
        85 works well empirically — catches "Apple Inc." ↔ "Apple" while
        rejecting false positives like "Amazon" ↔ "Amazon Web Services".
    """
    if not company_name or not company_name.strip():
        return None

    # Per-run in-memory cache
    if company_name in _session_cache:
        return _session_cache[company_name]

    metadata = _load_metadata()
    normalised = _normalise(company_name)

    # 1. Exact match after normalisation
    if normalised in metadata:
        result = metadata[normalised]
        _session_cache[company_name] = result
        return result

    # 2. Fuzzy match
    try:
        from rapidfuzz import process as rf_process, fuzz

        match = rf_process.extractOne(
            normalised,
            metadata.keys(),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        result = metadata[match[0]] if match else None
    except ImportError:
        # rapidfuzz not installed — fall back to no fuzzy match
        result = None

    _session_cache[company_name] = result
    return result


def clear_session_cache() -> None:
    """Clear the per-run cache. Call between ingest runs if needed."""
    _session_cache.clear()
