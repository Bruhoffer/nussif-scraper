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

from data_access import load_trades_df

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

@st.cache_data
def get_trades_data(days: int = 90) -> pd.DataFrame:
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

    # Derive unusual flag based on mid-point value
    df["Unusual"] = df["Mid Point"] > 100_000

    # Temporary party mapping (to be replaced with a senators metadata join)
    if "Party" not in df.columns:
        df["Party"] = "Unknown"

    # Temporary sector placeholder (until a ticker→sector mapping is added)
    if "Sector" not in df.columns:
        df["Sector"] = "Unknown"

    # Ensure Ticker column exists for filters; fall back to asset_name if needed
    if "Ticker" not in df.columns and "ticker" in df.columns:
        df["Ticker"] = df["ticker"].fillna("--")

    return df


df = get_trades_data(90)

# If there is no data in the DB yet, show a clear message instead of
# rendering empty charts/tables that can be confusing.
if df.empty:
    st.title("Capital Watch | Congress Trading Tracker")
    st.warning(
        "No PTR trades found in the database yet. "
        "Run the ingest script (python ingest_ptr_trades.py --days 90) "
        "and reload this app."
    )
    st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/congressman.png", width=80)
    st.title("Capital Watch")
    st.markdown("---")

    # Global Chamber filter; currently will just show ["Senate"], but is
    # ready for ["Senate", "House"] once House data is ingested.
    chambers = sorted(df["Chamber"].dropna().unique().tolist())
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
    st.info("Data refreshed every 24h via Azure Function.")

# Apply Chamber filter globally so all pages respect it
if selected_chambers:
    df = df[df["Chamber"].isin(selected_chambers)]
else:
    # If user deselects everything, show an empty dataset
    df = df.iloc[0:0]

# --- PAGE 1: EXECUTIVE DASHBOARD ---
if page == "Executive Dashboard":
    st.title("🏛️ Executive Dashboard")
    st.markdown("Overview of congressional trading activity over the last 90 days.")
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)

    total_vol = df["Mid Point"].sum()
    # Types in the DB are normalized to BUY/SELL/EXCHANGE by the scraper,
    # so we aggregate on those canonical values here.
    buy_vol = df[df["Type"] == "BUY"]["Mid Point"].sum()
    sell_vol = df[df["Type"] == "SELL"]["Mid Point"].sum()
    unusual_count = df[df["Unusual"] == True].shape[0]
    
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Volume (90D)</div><div class="metric-value">${total_vol/1e6:.1f}M</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Buy Volume</div><div class="metric-value" style="color:#10b981">${buy_vol/1e6:.1f}M</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Sell Volume</div><div class="metric-value" style="color:#ef4444">${sell_vol/1e6:.1f}M</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Unusual Trades</div><div class="metric-value" style="color:#f59e0b">{unusual_count}</div></div>', unsafe_allow_html=True)

    st.markdown("### Market Intelligence")
    
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Time Series
        time_df = df.groupby("Transaction Date")["Mid Point"].sum().reset_index()
        fig_time = px.area(time_df, x="Transaction Date", y="Mid Point", 
                         title="Daily Aggregate Trading Volume",
                         template="plotly_dark",
                         color_discrete_sequence=['#3b82f6'])
        fig_time.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_time, use_container_width=True)
        
    with c2:
        # Sector Pie
        sector_df = df.groupby("Sector")["Mid Point"].sum().reset_index()
        fig_sector = px.pie(sector_df, values="Mid Point", names="Sector", 
                          title="Sector Concentration",
                          template="plotly_dark",
                          hole=0.4)
        fig_sector.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sector, use_container_width=True)

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
        
    # Styling the dataframe (Phase 5, Step 21)
    st.dataframe(
        filtered_df.sort_values("Filing Date", ascending=False),
        column_config={
            "Mid Point": st.column_config.NumberColumn("Estimated Value", format="$%d"),
            "Unusual": st.column_config.CheckboxColumn("🚨"),
            "Ticker": st.column_config.TextColumn("Symbol", help="Stock Ticker"),
        },
        use_container_width=True,
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
    
    # Profile Header
    p_col1, p_col2 = st.columns([1, 3])
    with p_col1:
        st.image("https://img.icons8.com/fluency/144/user-male-circle.png", width=150)
    with p_col2:
        st.header(selected_senator)
        st.markdown(f"**Party:** {senator_df['Party'].iloc[0]}")
        st.markdown(f"**Total Estimated Volume (90D):** ${senator_df['Mid Point'].sum():,.2f}")
        st.markdown(f"**Most Traded Sector:** {senator_df['Sector'].mode()[0]}")

    st.markdown("---")
    
    # Senator Charts
    sc1, sc2 = st.columns(2)
    with sc1:
        ticker_counts = senator_df.groupby("Ticker")["Mid Point"].sum().sort_values(ascending=True).tail(10)
        fig_tickers = px.bar(ticker_counts, orientation='h', title="Top Holdings by Volume", 
                           template="plotly_dark", color_discrete_sequence=['#818cf8'])
        st.plotly_chart(fig_tickers, use_container_width=True)
        
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
        st.plotly_chart(fig_type, use_container_width=True)

    st.markdown("### Individual Transaction History")
    st.table(senator_df[["Transaction Date", "Ticker", "Type", "Amount Range", "Sector"]].head(10))

# --- FOOTER ---
st.markdown("---")
st.caption("Data provided for informational purposes only. Mid-point values are estimates based on legislative disclosure ranges.")