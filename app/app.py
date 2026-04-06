import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Ensure DATABASE_URL (and any other secrets) from .env are loaded before
# importing data_access/db.config, which constructs the SQLAlchemy engine.
load_dotenv()

from data_access import load_trades_df, load_volume_by_year_df, load_all_trades_df, load_portfolio_curve

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis_helpers import track_positions, compute_portfolio_curve, compute_portfolio_metrics

# --- CONFIGURATION ---
st.set_page_config(
    page_title="NUSSIF | Congress Trading Tracker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR PREMIUM UI ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* KPI Card Styling */
    .metric-card {
        background-color: #1e293b;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1e293b;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADER (REAL PTR TRADES FROM DB) ---

@st.cache_data(
    ttl=60 * 60,  # 60 minutes
    show_spinner="Loading latest PTR trades from database... contact @justin.cheong@u.nus.edu if this fails",
)
def get_trades_data(days: int = 365) -> pd.DataFrame:
    """Load and transform PTR trades from the database for the dashboard.

    This reshapes the raw DB columns to match the names expected by the
    existing UI so the rest of the app code can remain largely unchanged.
    """

    df = load_trades_df(days=days)

    if df.empty:
        return df

    # Rename DB columns to UI-friendly names
    df = df.rename(columns={
        "transaction_date": "Transaction Date",
        "filing_date": "Filing Date",
        "senator_display_name": "Senator",
        "transaction_type": "Type",
        "amount_range_raw": "Amount Range",
        "mid_point": "Mid Point",
        "chamber": "Chamber",
    })

    # Normalize Type so 'buy'/'sell' from House matches 'BUY'/'SELL' from Senate
    if "Type" in df.columns:
        df["Type"] = df["Type"].str.upper()

    # Expose Owner (member / spouse / joint) for downstream views
    if "owner" in df.columns and "Owner" not in df.columns:
        df["Owner"] = df["owner"].fillna("")

    # Derive unusual flag based on mid-point value
    df["Unusual"] = df["Mid Point"] > 100_000

    # Temporary party mapping (to be replaced with a senators metadata join)
    if "Party" not in df.columns:
        df["Party"] = "Unknown"

    # Sector handling: prefer any joined DB column `sector` (e.g. from
    # ticker_metadata) and expose it as the UI-facing `Sector` column.
    # Fallback to "Unknown" if we have no metadata yet so charts still
    # render.
    if "Sector" not in df.columns:
        if "sector" in df.columns:
            # Fill NaNs and replace empty strings (which Yahoo Finance often returns for ETFs)
            df["Sector"] = df["sector"].fillna("Unknown/ETF").replace("", "Unknown/ETF")
        else:
            df["Sector"] = "Unknown/ETF"

    # Expose price columns with UI-friendly names if present.
    if "price_at_transaction" in df.columns and "Price At Transaction" not in df.columns:
        df["Price At Transaction"] = df["price_at_transaction"]
    if "current_price" in df.columns and "Current Price" not in df.columns:
        df["Current Price"] = df["current_price"]

    # Ensure Ticker column exists for filters; fall back to asset_name if needed
    if "Ticker" not in df.columns and "ticker" in df.columns:
        df["Ticker"] = df["ticker"].fillna("--")
        
    # Filter out trades that have no valid ticker (e.g., "--" or empty)
    if "Ticker" in df.columns:
        df = df[~df["Ticker"].isin(["--", "", None])]

    # Calculate ROI/Profit for each trade with "First-Sell Closing" Heuristic
    if "Price At Transaction" in df.columns and "Current Price" in df.columns:
        # Force numeric types to handle any SQLite quirks
        df["Price At Transaction"] = pd.to_numeric(df["Price At Transaction"], errors='coerce')
        df["Current Price"] = pd.to_numeric(df["Current Price"], errors='coerce')
        
        # Sort values by transaction date to prepare for chronological matching
        df = df.sort_values("Transaction Date").reset_index(drop=True)
        
        # We will create an array to store the calculated ROI for each row
        rois = []
        
        for idx, row in df.iterrows():
            if row["Type"] != "BUY" or pd.isna(row["Price At Transaction"]) or row["Price At Transaction"] == 0:
                rois.append(np.nan)
                continue
                
            # For a BUY, look forward in time for the first SELL by the same Senator for the same Ticker
            future_sells = df.iloc[idx+1:]
            matching_sells = future_sells[
                (future_sells["Senator"] == row["Senator"]) & 
                (future_sells["Ticker"] == row["Ticker"]) & 
                (future_sells["Type"] == "SELL")
            ]
            
            if not matching_sells.empty:
                # Senator closed the position later! Use the sell price to calculate ROI
                sell_price = matching_sells.iloc[0]["Price At Transaction"]
                if pd.isna(sell_price) or sell_price == 0:
                    # Fallback to current price if sell price data is missing
                    calc_price = row["Current Price"]
                else:
                    calc_price = sell_price
            else:
                # Position is still open! Use Current Price
                calc_price = row["Current Price"]
                
            if pd.isna(calc_price):
                rois.append(np.nan)
            else:
                rois.append((calc_price - row["Price At Transaction"]) / row["Price At Transaction"])
                
        df["Estimated ROI (%)"] = pd.Series(rois) * 100
        # Force Mid Point to numeric as well just in case
        df["Mid Point"] = pd.to_numeric(df["Mid Point"], errors='coerce').fillna(0)
        df["Estimated Profit"] = (df["Estimated ROI (%)"] / 100) * df["Mid Point"]

    return df


@st.cache_data(ttl=60 * 60)
def get_volume_by_year_data() -> pd.DataFrame:
    """Load and prepare all-time trade data for the volume by year chart."""
    df = load_volume_by_year_df()
    if df.empty:
        return df

    df["Transaction Date"] = pd.to_datetime(df["transaction_date"])
    df["Year"] = df["Transaction Date"].dt.year
    df["Type"] = df["transaction_type"].str.upper()
    df["Mid Point"] = pd.to_numeric(df["mid_point"], errors="coerce")
    df["Senator"] = df["senator_display_name"]
    df["Chamber"] = df["chamber"]
    return df


@st.cache_data(ttl=60 * 60)
def get_all_trades_data() -> pd.DataFrame:
    """Load all historical trades (no date filter) for cross-year analysis."""
    df = load_all_trades_df()
    if df.empty:
        return df

    df = df.rename(columns={
        "transaction_date": "Transaction Date",
        "filing_date": "Filing Date",
        "senator_display_name": "Senator",
        "transaction_type": "Type",
        "amount_range_raw": "Amount Range",
        "mid_point": "Mid Point",
        "chamber": "Chamber",
    })
    if "Type" in df.columns:
        df["Type"] = df["Type"].str.upper()
    if "owner" in df.columns and "Owner" not in df.columns:
        df["Owner"] = df["owner"].fillna("")
    if "sector" in df.columns and "Sector" not in df.columns:
        df["Sector"] = df["sector"].fillna("Unknown/ETF").replace("", "Unknown/ETF")
    if "ticker" in df.columns and "Ticker" not in df.columns:
        df["Ticker"] = df["ticker"].fillna("--")
    if "price_at_transaction" in df.columns and "Price At Transaction" not in df.columns:
        df["Price At Transaction"] = pd.to_numeric(df["price_at_transaction"], errors="coerce")
    if "current_price" in df.columns and "Current Price" not in df.columns:
        df["Current Price"] = pd.to_numeric(df["current_price"], errors="coerce")

    # ROI / Estimated Profit — same first-sell-closing heuristic as get_trades_data()
    if "Price At Transaction" in df.columns and "Current Price" in df.columns:
        df = df.sort_values("Transaction Date").reset_index(drop=True)
        rois = []
        for idx, row in df.iterrows():
            if row["Type"] != "BUY" or pd.isna(row["Price At Transaction"]) or row["Price At Transaction"] == 0:
                rois.append(np.nan)
                continue
            future_sells = df.iloc[idx + 1:]
            matching_sells = future_sells[
                (future_sells["Senator"] == row["Senator"]) &
                (future_sells["Ticker"] == row["Ticker"]) &
                (future_sells["Type"] == "SELL")
            ]
            if not matching_sells.empty:
                sell_price = matching_sells.iloc[0]["Price At Transaction"]
                calc_price = sell_price if not pd.isna(sell_price) and sell_price != 0 else row["Current Price"]
            else:
                calc_price = row["Current Price"]
            if pd.isna(calc_price):
                rois.append(np.nan)
            else:
                rois.append((calc_price - row["Price At Transaction"]) / row["Price At Transaction"])
        df["Estimated ROI (%)"] = pd.Series(rois) * 100
        df["Mid Point"] = pd.to_numeric(df["Mid Point"], errors="coerce").fillna(0)
        df["Estimated Profit"] = (df["Estimated ROI (%)"] / 100) * df["Mid Point"]

    return df


df = get_trades_data(365)
vol_year_df = get_volume_by_year_data()
all_trades_df = get_all_trades_data()


# If there is no data in the DB yet, show a clear message instead of
# rendering empty charts/tables that can be confusing.
if df.empty:
    st.title("NUSSIF | Congress Trading Tracker")
    st.warning(
        "No PTR trades found in the database yet. "
        "Run the ingest script (python ingest_ptr_trades.py --days 365) "
        "and reload this app."
    )
    st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/congressman.png", width=80)
    st.title("NUSSIF")
    st.markdown("---")

    # Global Chamber filter; currently will just show ["Senate"], but is
    # ready for ["Senate", "House"] once House data is ingested.
    chambers = sorted(df["Chamber"].dropna().unique().tolist()) if "Chamber" in df.columns else []
    selected_chambers = st.multiselect(
        "Chamber",
        options=chambers,
        default=chambers,
    )

    page = st.radio(
        "Navigation",
        ["Executive Dashboard", "Live Intelligence Feed", "Senator Deep-Dives"],
        index=0
    )
    
    st.markdown("---")
    st.info("Data refreshed every 24h via GitHub Actions.")

# Apply Chamber filter globally so all pages respect it
if selected_chambers:
    df = df[df["Chamber"].isin(selected_chambers)]
else:
    # If user deselects everything, show an empty dataset
    df = df.iloc[0:0]

# --- PAGE 1: EXECUTIVE DASHBOARD ---
if page == "Executive Dashboard":
    st.title("🏛️ Executive Dashboard")
    st.markdown("Overview of congressional trading activity over the last 365 days.")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)

    total_vol = df["Mid Point"].sum()
    # Types in the DB are normalized to BUY/SELL/EXCHANGE by the scraper,
    # so we aggregate on those canonical values here.
    buy_vol = df[df["Type"] == "BUY"]["Mid Point"].sum()
    sell_vol = df[df["Type"] == "SELL"]["Mid Point"].sum()
    unusual_count = df[df["Unusual"] == True].shape[0]
    total_trades = len(df)

    # Additional summary stats
    buy_trades = df[df["Type"] == "BUY"]
    sell_trades = df[df["Type"] == "SELL"]
    buy_sell_ratio = (
        len(buy_trades) / len(sell_trades)
        if len(sell_trades) > 0
        else float("inf") if len(buy_trades) > 0
        else 0.0
    )

    vol_by_senator = df.groupby("Senator")["Mid Point"].sum()
    most_active_senator = vol_by_senator.idxmax() if not vol_by_senator.empty else "—"
    most_active_senator_vol = vol_by_senator.max() if not vol_by_senator.empty else 0.0
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Volume (365D)</div><div class="metric-value">${total_vol/1e6:.1f}M</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Buy Volume</div><div class="metric-value" style="color:#10b981">${buy_vol/1e6:.1f}M</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Sell Volume</div><div class="metric-value" style="color:#ef4444">${sell_vol/1e6:.1f}M</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Unusual Trades</div><div class="metric-value" style="color:#f59e0b">{unusual_count}</div></div>', unsafe_allow_html=True)

    # Second KPI row focused on behavior and concentration
    kpi2_col1, kpi2_col2, kpi2_col3, kpi2_col4 = st.columns(4)
    with kpi2_col1:
        ratio_text = "∞" if buy_sell_ratio == float("inf") else f"{buy_sell_ratio:.2f}x"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Buy / Sell Trade Count</div>'
            f'<div class="metric-value">{ratio_text}</div></div>',
            unsafe_allow_html=True,
        )

    with kpi2_col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Trades (365D)</div>'
            f'<div class="metric-value">{total_trades:,}</div></div>',
            unsafe_allow_html=True,
        )

    with kpi2_col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Most Active (Vol)</div>'
            f'<div class="metric-value" style="font-size: 1.2rem; font-weight: 800;">{most_active_senator}</div></div>',
            unsafe_allow_html=True,
        )

    # Placeholder for future KPI (e.g. most active party or sector)
    with kpi2_col4:
        top_sector = (
            df.groupby("Sector")["Mid Point"].sum().idxmax()
            if "Sector" in df.columns and not df["Sector"].empty
            else "—"
        )
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Most Traded Sector (Vol)</div>'
            f'<div class="metric-value">{top_sector}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Market Intelligence")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Time Series
        # Filter for the last 365 days of Transaction Dates to keep the x-axis clean
        cutoff_date = (pd.to_datetime('today') - pd.DateOffset(days=365)).date()
        # Convert to datetime series to support comparison
        transaction_dates = pd.to_datetime(df["Transaction Date"]).dt.date
        recent_df = df[transaction_dates >= cutoff_date]
        
        time_df = recent_df.groupby("Transaction Date")["Mid Point"].sum().reset_index()
        fig_time = px.area(time_df, x="Transaction Date", y="Mid Point", 
                         title="Daily Aggregate Trading Volume (Past Year)",
                         template="plotly_dark",
                         color_discrete_sequence=['#3b82f6'])
        fig_time.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_time, width='stretch')
        
    with c2:
        # Sector Pie
        sector_df = df.groupby("Sector")["Mid Point"].sum().reset_index()
        fig_sector = px.pie(sector_df, values="Mid Point", names="Sector", 
                          title="Sector Concentration",
                          template="plotly_dark",
                          hole=0.4)
        fig_sector.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sector, width='stretch')

    # Top stocks by volume section – allows you to see which tickers
    # are attracting the most congressional flow, and whether that
    # flow is net buying or selling.
    st.markdown("### Top Stocks by Volume")

    metric_mode = st.radio(
        "Metric",
        ["Buy Volume", "Sell Volume", "Net Volume (Buy - Sell)"],
        horizontal=True,
        key="top_stocks_metric",
    )

    buy_df = df[df["Type"] == "BUY"]
    sell_df = df[df["Type"] == "SELL"]

    buy_vol_by_ticker = buy_df.groupby("Ticker")["Mid Point"].sum()
    sell_vol_by_ticker = sell_df.groupby("Ticker")["Mid Point"].sum()

    vol_df = pd.DataFrame({
        "Buy Volume": buy_vol_by_ticker,
        "Sell Volume": sell_vol_by_ticker,
    }).fillna(0.0)

    vol_df["Net Volume (Buy - Sell)"] = vol_df["Buy Volume"] - vol_df["Sell Volume"]

    sort_col = metric_mode

    top_stocks = (
        vol_df.sort_values(sort_col, ascending=False)
        .head(20)
        .reset_index(names="Ticker")
    )

    fig_top_stocks = px.bar(
        top_stocks,
        x="Ticker",
        y=sort_col,
        title=f"Top Stocks by {sort_col} (365D)",
        template="plotly_dark",
        color_discrete_sequence=['#3b82f6'],
    )
    fig_top_stocks.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Ticker",
        yaxis_title=sort_col,
    )
    st.plotly_chart(fig_top_stocks, width='stretch')

    st.markdown("#### Stock Volume Leaderboard")
    st.dataframe(
        vol_df.sort_values(sort_col, ascending=False),
        column_config={
            "Buy Volume": st.column_config.NumberColumn("Buy Volume", format="$%d"),
            "Sell Volume": st.column_config.NumberColumn("Sell Volume", format="$%d"),
            "Net Volume (Buy - Sell)": st.column_config.NumberColumn(
                "Net Volume (Buy - Sell)", format="$%d"
            ),
        },
        width='stretch',
    )
    
    # --- Trade Volume by Year ---
    st.markdown("### Trade Volume by Year")

    if not vol_year_df.empty:
        # Apply the same Chamber filter
        year_filtered = vol_year_df[vol_year_df["Chamber"].isin(selected_chambers)] if selected_chambers else vol_year_df

        year_agg = (
            year_filtered[year_filtered["Type"].isin(["BUY", "SELL"])]
            .groupby(["Year", "Type"])["Mid Point"]
            .sum()
            .reset_index()
        )

        fig_year = px.bar(
            year_agg,
            x="Year",
            y="Mid Point",
            color="Type",
            barmode="group",
            title="Annual Trade Volume — Buy vs Sell",
            template="plotly_dark",
            color_discrete_map={"BUY": "#10b981", "SELL": "#ef4444"},
        )
        fig_year.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickmode="linear", dtick=1),
            yaxis_title="Volume (Mid Point $)",
        )
        st.plotly_chart(fig_year, width="stretch")

    # Leaderboard of Senator Average ROI
    if "Estimated ROI (%)" in df.columns:
        st.markdown("## Top Politicians by ROI (Copy-Trade Strategy)")
        st.markdown("Return if you had copied every single BUY trade this politician made over the period.")
        
        # Only evaluate ROI based on buy transactions
        buy_df = df[df["Type"] == "BUY"].copy()
        
        # Flag if trade is positive
        buy_df["Is_Positive"] = buy_df["Estimated ROI (%)"] > 0
        
        senator_roi = buy_df.groupby("Senator").agg(
            Avg_ROI=("Estimated ROI (%)", "mean"),
            Trades=("Ticker", "count"),
            Positive_Trades=("Is_Positive", "sum")
        ).reset_index()
        
        # Filter for senators with at least 1 buy trade
        senator_roi = senator_roi[senator_roi["Trades"] > 0]
        
        # Calculate Hit Rate
        senator_roi["Hit_Rate"] = (senator_roi["Positive_Trades"] / senator_roi["Trades"]) * 100
        
        # Drop the intermediate column to clean up the display
        senator_roi = senator_roi.drop(columns=["Positive_Trades"])
        
        senator_roi = senator_roi.sort_values("Avg_ROI", ascending=False).reset_index(drop=True)
        
        st.dataframe(
            senator_roi,
            column_config={
                "Senator": "Legislator",
                "Avg_ROI": st.column_config.NumberColumn("Average Copy-Trade ROI", format="%.2f%%"),
                "Hit_Rate": st.column_config.NumberColumn("Hit Rate", format="%.2f%%"),
                "Trades": st.column_config.NumberColumn("Total Buy Trades")
            },
            width='stretch',
            hide_index=True
        )

# --- PAGE 2: LIVE INTELLIGENCE FEED ---
elif page == "Live Intelligence Feed":
    st.title("📡 Live Intelligence Feed")
    
    # Advanced Filters
    with st.expander("🔍 Filter Controls", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            # Use whatever party labels actually exist in the data so that
            # the default selection does not accidentally filter out all rows
            # (e.g. when everything is currently "Unknown").
            parties = sorted(df["Party"].dropna().unique().tolist())
            party_filter = st.multiselect(
                "Filter by Party",
                options=parties,
                default=parties,
            )
        with f_col2:
            ticker_search = st.text_input("Search Ticker", placeholder="e.g. NVDA")
        with f_col3:
            unusual_only = st.checkbox("Show Unusual Trades Only (> $100k Mid-point)")
            
    # Apply logic
    filtered_df = df[df["Party"].isin(party_filter)]
    if ticker_search:
        filtered_df = filtered_df[filtered_df["Ticker"].str.contains(ticker_search.upper())]
    if unusual_only:
        filtered_df = filtered_df[filtered_df["Unusual"] == True]
        
    # Select columns to display
    display_cols = ["Filing Date", "Transaction Date", "Senator", "Ticker", "Type", "Amount Range", "Mid Point", "Unusual"]
    if "Estimated Profit" in df.columns and "Estimated ROI (%)" in df.columns:
        display_cols.extend(["Price At Transaction", "Current Price", "Estimated Profit", "Estimated ROI (%)"])

    # Make sure all columns exist
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    
    # Styling the dataframe (Phase 5, Step 21)
    st.dataframe(
        filtered_df[display_cols].sort_values("Filing Date", ascending=False),
        column_config={
            "Mid Point": st.column_config.NumberColumn("Estimated Value", format="$%d"),
            "Unusual": st.column_config.CheckboxColumn("🚨"),
            "Ticker": st.column_config.TextColumn("Symbol", help="Stock Ticker"),
            "Price At Transaction": st.column_config.NumberColumn("Entry/Exit Price", format="$%.2f"),
            "Current Price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
            "Estimated Profit": st.column_config.NumberColumn("Est. Profit", format="$%d"),
            "Estimated ROI (%)": st.column_config.NumberColumn("ROI", format="%.2f%%"),
        },
        width='stretch',
        hide_index=True
    )
    
    st.download_button(
        label="📥 Export to CSV",
        data=filtered_df.to_csv().encode('utf-8'),
        file_name='congress_trades.csv',
        mime='text/csv',
    )

# --- PAGE 3: SENATOR DEEP-DIVES ---
elif page == "Senator Deep-Dives":
    st.title("👤 Senator Profiles")
    
    selected_senator = st.selectbox("Select a Legislator to Analyze", options=df["Senator"].unique())
    
    senator_df = df[df["Senator"] == selected_senator]

    # Split into buys and sells for clearer volume attribution
    senator_buys = senator_df[senator_df["Type"] == "BUY"]
    senator_sells = senator_df[senator_df["Type"] == "SELL"]
    
    # Profile Header
    p_col1, p_col2 = st.columns([1, 3])
    with p_col1:
        st.image("https://img.icons8.com/fluency/144/user-male-circle.png", width=150)
    with p_col2:
        st.header(selected_senator)
        st.markdown(f"**Party:** {senator_df['Party'].iloc[0]}")
        st.markdown(f"**Total Estimated Volume (365D):** ${senator_df['Mid Point'].sum():,.2f}")
        
        if "Estimated Profit" in senator_df.columns:
            sen_profit = senator_df["Estimated Profit"].sum()
            prof_color = "green" if sen_profit >= 0 else "red"
            st.markdown(f"**Total Estimated Profit (365D):** <span style='color:{prof_color}'>${sen_profit:,.2f}</span>", unsafe_allow_html=True)
            
        st.markdown(f"**Most Traded Sector:** {senator_df['Sector'].mode()[0] if not senator_df['Sector'].empty else 'Unknown'}")

    st.markdown("---")
    
    # Senator Charts
    sc1, sc2 = st.columns(2)
    with sc1:
        # Only count BUY trades when attributing "Top Holdings" to avoid
        # inflating exposure with sells/trim activity.
        buy_ticker_vol = (
            senator_buys.groupby("Ticker")["Mid Point"]
            .sum()
            .sort_values(ascending=True)
            .tail(10)
        )
        fig_tickers = px.bar(
            buy_ticker_vol,
            orientation='h',
            title="Top Buy Holdings by Volume",
            template="plotly_dark",
            color_discrete_sequence=['#10b981'],
        )
        st.plotly_chart(fig_tickers, width='stretch')
        
    with sc2:
        type_counts = senator_df.groupby("Type").size().reset_index(name="count")
        fig_type = px.pie(
            type_counts,
            values="count",
            names="Type",
            title="Buy vs Sell Sentiment",
            template="plotly_dark",
            # Match normalized BUY/SELL labels produced by the scraper.
            color_discrete_map={"BUY": "#10b981", "SELL": "#ef4444"},
        )
        st.plotly_chart(fig_type, width='stretch')

    # Trade Volume by Year for this senator
    st.markdown("### Trade Volume by Year")
    senator_year_df = vol_year_df[vol_year_df["Senator"] == selected_senator] if not vol_year_df.empty else pd.DataFrame()

    if not senator_year_df.empty:
        senator_year_agg = (
            senator_year_df[senator_year_df["Type"].isin(["BUY", "SELL"])]
            .groupby(["Year", "Type"])["Mid Point"]
            .sum()
            .reset_index()
        )
        fig_sen_year = px.bar(
            senator_year_agg,
            x="Year",
            y="Mid Point",
            color="Type",
            barmode="group",
            title=f"{selected_senator} — Annual Trade Volume",
            template="plotly_dark",
            color_discrete_map={"BUY": "#10b981", "SELL": "#ef4444"},
        )
        fig_sen_year.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickmode="linear", dtick=1),
            yaxis_title="Volume (Mid Point $)",
        )
        st.plotly_chart(fig_sen_year, width="stretch")
    else:
        st.info("No annual volume data available for this legislator.")

    # --- Current Holdings + Portfolio Performance ---
    senator_all_df = all_trades_df[all_trades_df["Senator"] == selected_senator] if not all_trades_df.empty else pd.DataFrame()
    open_positions, closed_trades = track_positions(senator_all_df) if not senator_all_df.empty else (pd.DataFrame(), pd.DataFrame())

    st.markdown("### Current Holdings")
    st.caption(
        "Positions tracked using a stateful model: BUY opens/adds to long, "
        "SELL closes long or opens short, and vice versa. "
        "Share counts are estimated from mid_point ÷ price_at_transaction."
    )

    if open_positions.empty:
        st.info("No open positions found for this legislator.")
    else:
        # Only include positions where current price was fetched, so all three
        # KPIs are computed over the same subset and remain internally consistent.
        priced = open_positions[open_positions["Current Price"].notna()]
        total_current_value = priced["Current Value"].sum(skipna=True)
        total_cost_basis = priced["Cost Basis"].sum(skipna=True)
        total_unrealized_pnl = priced["Unrealized P&L"].sum(skipna=True)

        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Est. Portfolio Value</div>'
                f'<div class="metric-value">${total_current_value:,.0f}</div></div>',
                unsafe_allow_html=True,
            )
        with h_col2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Total Cost Basis</div>'
                f'<div class="metric-value">${total_cost_basis:,.0f}</div></div>',
                unsafe_allow_html=True,
            )
        with h_col3:
            pnl_color = "#10b981" if total_unrealized_pnl >= 0 else "#ef4444"
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Total Unrealized P&L</div>'
                f'<div class="metric-value" style="color:{pnl_color}">${total_unrealized_pnl:,.0f}</div></div>',
                unsafe_allow_html=True,
            )

        st.dataframe(
            open_positions,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Sector": st.column_config.TextColumn("Sector"),
                "Direction": st.column_config.TextColumn("Direction"),
                "Shares (Est)": st.column_config.NumberColumn("Shares (Est)", format="%.1f"),
                "Avg Entry Price": st.column_config.NumberColumn("Avg Entry Price", format="$%.2f"),
                "Current Price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
                "Cost Basis": st.column_config.NumberColumn("Cost Basis", format="$%d"),
                "Current Value": st.column_config.NumberColumn("Current Value", format="$%d"),
                "Unrealized P&L": st.column_config.NumberColumn("Unrealized P&L", format="$%d"),
                "ROI (%)": st.column_config.NumberColumn("ROI", format="%.2f%%"),
                "Opened Date": st.column_config.DateColumn("Opened Date"),
                "Last Trade Date": st.column_config.DateColumn("Last Trade Date"),
            },
            hide_index=True,
            width="stretch",
        )

    # --- Portfolio Performance Tracker ---
    st.markdown("### Portfolio Performance")
    st.caption(
        "Estimated portfolio value over time. Precomputed weekly — "
        "run `python -m ingest.portfolio_snapshots` to populate for the first time."
    )

    # Try DB-backed curve first (fast). Fall back to live computation only if
    # no snapshots exist yet (first run before weekly ingest has executed).
    curve_df = load_portfolio_curve(selected_senator)
    if curve_df.empty and not senator_all_df.empty:
        st.info("No precomputed curve found. Computing live — this may take a minute...")
        with st.spinner("Building portfolio curve (fetching historical prices)..."):
            curve_df = compute_portfolio_curve(senator_all_df)

    if curve_df.empty or curve_df["portfolio_value"].sum() == 0:
        st.info("Not enough data to build a portfolio curve for this legislator.")
    else:
        # Portfolio curve chart
        fig_curve = px.area(
            curve_df,
            x="date",
            y="portfolio_value",
            title=f"{selected_senator} — Estimated Portfolio Value Over Time",
            template="plotly_dark",
            color_discrete_sequence=["#3b82f6"],
        )
        fig_curve.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Portfolio Value ($)",
            xaxis_title="Date",
        )

        # Overlay buy/sell events as markers
        if not senator_all_df.empty:
            events = senator_all_df[senator_all_df["Type"].isin(["BUY", "SELL"])].copy()
            events["Transaction Date"] = pd.to_datetime(events["Transaction Date"])
            buy_events = events[events["Type"] == "BUY"]
            sell_events = events[events["Type"] == "SELL"]

            fig_curve.add_scatter(
                x=buy_events["Transaction Date"], y=[0] * len(buy_events),
                mode="markers", marker=dict(color="#10b981", size=8, symbol="triangle-up"),
                name="BUY",
            )
            fig_curve.add_scatter(
                x=sell_events["Transaction Date"], y=[0] * len(sell_events),
                mode="markers", marker=dict(color="#ef4444", size=8, symbol="triangle-down"),
                name="SELL",
            )

        st.plotly_chart(fig_curve, width="stretch")

        # Risk/return metrics
        metrics = compute_portfolio_metrics(curve_df, closed_trades)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        def _fmt_metric(val, fmt):
            return fmt.format(val) if val is not None else "N/A"

        with m_col1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Max Drawdown</div>'
                f'<div class="metric-value" style="color:#ef4444">{_fmt_metric(metrics["max_drawdown_pct"], "{:.1f}%")}</div></div>',
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Beta (vs SPY)</div>'
                f'<div class="metric-value">{_fmt_metric(metrics["beta"], "{:.2f}")}</div></div>',
                unsafe_allow_html=True,
            )
        with m_col3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div>'
                f'<div class="metric-value">{_fmt_metric(metrics["sharpe"], "{:.2f}")}</div></div>',
                unsafe_allow_html=True,
            )
        with m_col4:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Win Rate (Closed)</div>'
                f'<div class="metric-value">{_fmt_metric(metrics["win_rate_pct"], "{:.1f}%")}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Individual Transaction History")

    history_cols = [
        "Filing Date",
        "Transaction Date",
        "Ticker",
        "Type",
        "Amount Range",
        "Mid Point",
        "Owner",
        "Sector",
    ]
    
    if "Estimated Profit" in senator_all_df.columns and "Estimated ROI (%)" in senator_all_df.columns:
        history_cols.extend(["Price At Transaction", "Current Price", "Estimated Profit", "Estimated ROI (%)"])

    # Use senator_all_df so Congress API trades (pre-2024) are also visible,
    # not just the Senate PTR filings loaded into the 365-day senator_df.
    available_cols = [c for c in history_cols if c in senator_all_df.columns]
    history_df = senator_all_df[available_cols].sort_values(
        ["Filing Date", "Transaction Date"], ascending=[False, False]
    )

    st.dataframe(
        history_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Mid Point": st.column_config.NumberColumn("Estimated Value", format="$%d"),
            "Price At Transaction": st.column_config.NumberColumn("Entry/Exit Price", format="$%.2f"),
            "Current Price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
            "Estimated Profit": st.column_config.NumberColumn("Est. Profit", format="$%d"),
            "Estimated ROI (%)": st.column_config.NumberColumn("ROI", format="%.2f%%"),
        },
    )

# --- FOOTER ---
st.markdown("---")
st.caption("Data provided for informational purposes only. Mid-point values are estimates based on legislative disclosure ranges.")