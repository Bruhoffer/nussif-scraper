import requests
import pandas as pd
import time
import json
import re
import numpy as np

import sys
import os
# Ensure the root of the project is in the Python path so we can import from `db`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()
from db.config import init_db
from db.upsert import upsert_trades
from db.prices import enrich_prices_for_trades, update_all_current_prices

# --- CONFIGURATION ---
API_KEY = os.getenv('rapid-api-key')
API_HOST = "politician-trade-tracker1.p.rapidapi.com"

headers = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

def get_all_politician_names():
    """Fetches the directory of all politicians available in the API."""
    url = f"https://{API_HOST}/get_politicians"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # Returns a dict where keys are politician names
        return list(response.json().keys())
    return []

def parse_amount(amount_str):
    """
    Parses a string like '1K-15K', '50K-100K', '100K-250K', or '>1M'
    into (amount_min, amount_max, mid_point).
    """
    if pd.isna(amount_str) or not isinstance(amount_str, str):
        return None, None, None

    amount_str = amount_str.strip()
    
    # Handle >1M case
    if ">" in amount_str:
        num = float(re.sub(r'[^\d.]', '', amount_str))
        if "M" in amount_str:
            num *= 1_000_000
        elif "K" in amount_str:
            num *= 1_000
        return num, None, num

    parts = amount_str.split('-')
    if len(parts) != 2:
        return None, None, None
    
    def parse_part(p):
        p = p.strip()
        val = float(re.sub(r'[^\d.]', '', p))
        if 'M' in p:
            val *= 1_000_000
        elif 'K' in p:
            val *= 1_000
        return val

    try:
        min_val = parse_part(parts[0])
        max_val = parse_part(parts[1])
        mid_val = (min_val + max_val) / 2
        return min_val, max_val, mid_val
    except Exception:
        return None, None, None

def transform_for_db(df):
    """
    Transforms the rapidapi congress trade dataframe into a format 
    compatible with the existing Trade DB model.
    """
    if df.empty:
        return pd.DataFrame()
        
    db_df = pd.DataFrame()
    
    # Map straightforward fields
    db_df['senator_name'] = df['politician_name']
    
    # Split the name
    db_df['senator_first_name'] = df['politician_name'].apply(lambda x: x.split()[0] if isinstance(x, str) else None)
    db_df['senator_last_name'] = df['politician_name'].apply(lambda x: " ".join(x.split()[1:]) if isinstance(x, str) and len(x.split()) > 1 else None)
    db_df['senator_display_name'] = df['politician_name']
    
    db_df['chamber'] = df['chamber']
    
    db_df['asset_name'] = df['company']
    
    # Clean up ticker (e.g. 'FCN:US' -> 'FCN', 'N/A' -> None)
    db_df['ticker'] = df['ticker'].apply(
        lambda x: x.split(':')[0] if isinstance(x, str) and x != 'N/A' else None
    )
    
    db_df['transaction_date'] = pd.to_datetime(df['trade_date'], errors='coerce').dt.date
    
    # Estimate filing date if possible
    db_df['filing_date'] = db_df['transaction_date'] + pd.to_timedelta(pd.to_numeric(df['days_until_disclosure'], errors='coerce'), unit='D')
    
    db_df['transaction_type'] = df['trade_type']
    db_df['transaction_type_raw'] = df['trade_type']
    
    db_df['amount_range_raw'] = df['trade_amount']
    
    # Parse amounts
    parsed_amounts = df['trade_amount'].apply(parse_amount)
    db_df['amount_min'] = parsed_amounts.apply(lambda x: x[0])
    db_df['amount_max'] = parsed_amounts.apply(lambda x: x[1])
    db_df['mid_point'] = parsed_amounts.apply(lambda x: x[2])
    
    # Fields missing from API that exist in DB
    db_df['report_id'] = None
    db_df['report_type'] = None
    db_df['report_format'] = None
    db_df['owner'] = None
    db_df['asset_type'] = None
    db_df['comment'] = None
    
    return db_df

def get_full_profile(name):
    """Fetches the complete historical trade data for a specific person."""
    url = f"https://{API_HOST}/get_profile"
    params = {"name": name}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def run_pipeline(limit_politicians=10):
    """
    Runs the full data pull. 
    limit_politicians: Set to a small number to test, or None for ALL.
    """
    print("📋 Fetching politician directory...")
    names = get_all_politician_names()
    
    if limit_politicians:
        names = names[:limit_politicians]
    
    print(f"🔎 Found {len(names)} politicians. Starting deep-pull...")
    
    all_trades = []
    
    for i, name in enumerate(names):
        print(f"[{i+1}/{len(names)}] Pulling history for: {name}")
        profile = get_full_profile(name)
        
        if profile and "Trade Data" in profile:
            # The 'Trade Data' key in a profile is a list of trades
            trades = profile["Trade Data"]
            for trade_info in trades:
                # Add the politician's name to the trade record
                trade_info['politician_name'] = name
                all_trades.append(trade_info)
        
        # Rate limit protection: pause for 1 second between requests
        time.sleep(1)

    # Convert to DataFrame
    df = pd.DataFrame(all_trades)
    
    if not df.empty:
        # Clean up: Convert trade_date to actual datetime objects
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        df = df.sort_values(by='trade_date', ascending=False)
        
        # Save to CSV for backup
        filename = f"congress_trades_{time.strftime('%Y%m%d')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ SUCCESS! {len(df)} trades saved to {filename}")
        
        # --- DB Ingestion ---
        print("\n🚀 Transforming and ingesting data into the database...")
        db_df = transform_for_db(df)
        
        # Initialize the DB
        init_db()
        
        # Enrich prices before DB insert to avoid missing historical info
        print("Enriching prices for new trades...")
        failed_tickers, failed_pairs = enrich_prices_for_trades(db_df)
        if failed_tickers or failed_pairs:
            print(f"Pricing missing for {len(failed_tickers)} tickers and {len(failed_pairs)} pairs.")
            
        # Replace NaN with None (SQL NULL) and format floats
        db_df = db_df.replace({np.nan: None, float("inf"): None, float("-inf"): None})
        price_cols = ["price_at_transaction", "current_price"]
        for col in price_cols:
            if col in db_df.columns:
                db_df[col] = db_df[col].apply(lambda x: round(x, 4) if pd.notnull(x) else None)
        
        db_df = db_df.astype(object)
        db_df = db_df.where(pd.notnull(db_df), None)
        
        # Convert df to dict for upsert
        trades_to_insert = db_df.to_dict(orient="records")
        inserted = upsert_trades(trades_to_insert)
        
        print(f"Upsert complete. Inserted {inserted} new trades (scraped {len(trades_to_insert)} total).")
        
        # Finally, update current prices for ALL trades in DB (including our new House ones)
        print("Updating current prices for all historical trades in the database...")
        update_all_current_prices()
        
        return db_df
    else:
        print("\n❌ No trade data was found.")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape ALL Congress trades via API and upsert to DB")
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit number of politicians to scrape. If not provided, scrapes ALL."
    )
    args = parser.parse_args()

    # Note: Running for ALL politicians will take ~5-10 minutes due to rate limits.
    final_df = run_pipeline(limit_politicians=args.limit)
    
    if final_df is not None:
        print("\n--- Preview of Data ---")
        print(final_df[['senator_name', 'asset_name', 'transaction_type', 'amount_range_raw']].head(10))
