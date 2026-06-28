# nussif-scraper — Congressional Insider Trade Tracker

An automated data pipeline that scrapes, ingests, and analyses US congressional **Periodic Transaction Reports (PTRs)** — the mandatory disclosures of stock trades made by members of Congress.

Built for the NUS Student Investment Fund (NUSSIF).

---

## What It Does

Congress members are required to disclose stock trades within 45 days under the STOCK Act. This system:

1. **Scrapes** the House Financial Disclosures portal for new PTR filings
2. **Ingests** each trade into a SQL database with normalised schema
3. **Backfills** historical data (up to 5 years)
4. **Analyses** trading patterns — by legislator, sector, ticker, timing relative to committee assignments
5. **Visualises** results in a Streamlit dashboard

Daily and weekly scheduled runs keep the database current.

---

## Architecture

```
scraper/              Scrapes the House Disclosure portal for new PTR filings
ingest/               Normalises raw filing data and writes to the database
insiderscraper/       Core scraper logic (shared module)
app/
  app.py              Streamlit dashboard
  data_access.py      DB query layer for the app
analysis_helpers.py   Quantitative analysis functions (by ticker, sector, legislator)
backfill.py           Backfill recent PTR filings
backfill_ptr_5y.py    Bulk backfill of 5 years of historical PTRs
ingest_ptr_trades.py  Ingest parsed PTR trades to the DB
ptr_batch_to_csv.py   Export batch PTRs to CSV
run_daily.py          Scheduled daily run (new filings)
run_weekly.py         Scheduled weekly run (broader refresh + analysis)
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Scraping | BeautifulSoup · requests · rapidfuzz (fuzzy name matching) |
| Database | SQLAlchemy · SQL Server (pyodbc) |
| Analysis | pandas · numpy |
| Dashboard | Streamlit · Plotly |
| Finance data | yfinance (price correlation) |

---

## Setup

**1. Clone and install**

```bash
git clone https://github.com/Bruhoffer/nussif-scraper.git
cd nussif-scraper
pip install -r requirements.txt
```

**2. Environment variables** — create a `.env` file:

```env
DB_CONNECTION_STRING=mssql+pyodbc://...
```

**3. Backfill historical data** (optional, runs once)

```bash
python backfill_ptr_5y.py
```

**4. Run the dashboard**

```bash
streamlit run app/app.py
```

**5. Schedule recurring ingestion**

```bash
# Daily (new filings)
python run_daily.py

# Weekly (full refresh)
python run_weekly.py
```

---

## Data Source

Congressional PTR filings are publicly available at [disclosures.house.gov](https://disclosures.house.gov/). The scraper navigates the filing search, downloads individual reports, parses trade entries, and fuzzy-matches legislator names.

A sample dataset (`congress_trades_20260310.csv`) is included with ~5,000 trades through March 2026.

---

## Analysis Highlights

`analysis_helpers.py` provides functions for:

- Aggregate trade volume by legislator and party
- Sector and ticker frequency (most-traded stocks in Congress)
- Trade timing analysis (relative to committee assignments and legislation)
- Price performance of congressional trades vs. S&P 500 benchmark
