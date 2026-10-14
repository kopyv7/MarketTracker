import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import urllib.request
import xml.etree.ElementTree as ET

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Market Tracker 2.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_ticker_from_search(query):
    """Searches Yahoo Finance for a company name and returns the ticker symbol."""
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        if 'quotes' in data and len(data['quotes']) > 0:
            for item in data['quotes']:
                # Find the first real stock/ETF match
                if item.get('quoteType') in ['EQUITY', 'ETF', 'INDEX', 'MUTUALFUND']:
                    return item['symbol'], item.get('shortname', item['symbol'])
    except Exception:
        pass
    return query.upper(), query.upper()

def get_smart_prediction(df):
    """Calculates technical indicators to recommend BUY, SELL, or KEEP."""
    if df is None or len(df) < 50:
        return "KEEP 🟡", "Not enough historical data to run the predictive algorithm."

    # Calculate 50-day Simple Moving Average (SMA)
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate 14-day Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # Avoid division by zero by replacing 0s with tiny numbers
    loss = loss.replace(0, 0.00001) 
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Get the most recent values
    current_price = df['Close'].iloc[-1]
    current_sma = df['SMA_50'].iloc[-1]
    current_rsi = df['RSI'].iloc[-1]
    
    # Algorithm Logic
    if current_rsi < 30:
        action = "BUY 🟢"
        reason = f"The 14-day RSI is severely oversold at **{current_rsi:.1f}**. The algorithm predicts this asset is currently undervalued and presents a strong buying opportunity for a bounce back."
    elif current_rsi > 70:
        action = "SELL 🔴"
        reason = f"The 14-day RSI is highly overbought at **{current_rsi:.1f}**. The asset has run up quickly, meaning buyers are exhausted and it may be due for a pullback. Consider taking profits."
    elif current_price > current_sma:
        action = "KEEP (Bullish) 🟢"
        reason = f"The current price (${current_price:.2f}) is trending strong above its 50-day average (${current_sma:.2f}). RSI is neutral ({current_rsi:.1f}). The algorithm recommends holding to ride the upward momentum."
    else:
        action = "KEEP (Caution) 🟡"
        reason = f"The asset is weak, trending below its 50-day average (${current_sma:.2f}). However, the RSI is neutral ({current_rsi:.1f}), meaning there is no need to panic sell. The algorithm recommends waiting for a clear reversal before buying more."
        
    return action, reason

@st.cache_data(ttl=300)
def fetch_market_data(symbol, period="1y"):
    """Fetches historical market data and guarantees news via failsafes."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1y") 
        
        if df.empty:
            return None, [], {}, symbol
            
        info = {}
        try:
            info = ticker.info
        except:
            pass
            
        # 1. Try standard yfinance news
        news = ticker.news[:5] if hasattr(ticker, 'news') else []
        
        # 2. FAILSAFE: If standard news is empty, scrape Yahoo RSS directly
        if not news:
            try:
                url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=3) as response:
                    xml_data = response.read()
                    root = ET.fromstring(xml_data)
                    for item in root.findall('.//item')[:5]:
                        news.append({
                            'title': item.find('title').text,
                            'link': item.find('link').text,
                            'publisher': 'Yahoo Finance RSS'
                        })
            except:
                pass
                
        # 3. ULTIMATE FAILSAFE: If offline or blocked, generate a guaranteed fallback link
        if not news:
            news = [{
                'title': f"Latest Live Market Updates for {symbol}",
                'link': f"https://finance.yahoo.com/quote/{symbol}/news",
                'publisher': "Yahoo Finance Network"
            }]

        name = info.get('shortName', symbol)
        return df, news, info, name
    except Exception:
        return None, [], {}, symbol

def create_chart(df, period_key):
    """Creates a beautiful interactive Apple-Stocks style chart."""
    now = df.index.max()
    if period_key == "1W": start_date = now - pd.Timedelta(days=7)
    elif period_key == "1M": start_date = now - pd.Timedelta(days=30)
    elif period_key == "3M": start_date = now - pd.Timedelta(days=90)
    elif period_key == "6M": start_date = now - pd.Timedelta(days=180)
    elif period_key == "YTD": start_date = pd.to_datetime(f"{now.year}-01-01").tz_localize(df.index.tz)
    elif period_key == "1Y": start_date = now - pd.Timedelta(days=365)
    else: start_date = df.index.min()

    chart_df = df[df.index >= start_date]
    if chart_df.empty: chart_df = df

    # Determine color (Green if up, Red if down)
    start_price = chart_df['Close'].iloc[0]
    end_price = chart_df['Close'].iloc[-1]
    line_color = "#34C759" if end_price >= start_price else "#FF3B30"
    
    # Calculate bounds for extreme zooming (removes the flat line problem)
    min_price = chart_df['Low'].min()
    max_price = chart_df['High'].max()
    padding = (max_price - min_price) * 0.05
    if padding == 0: padding = start_price * 0.01 
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df.index, 
        y=chart_df['Close'],
        mode='lines',
        line=dict(color=line_color, width=2),
        fill='tozeroy',
        fillcolor=f"rgba({52 if line_color=='#34C759' else 255}, {199 if line_color=='#34C759' else 59}, {89 if line_color=='#34C759' else 48}, 0.1)"
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=True, visible=True),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)', 
            side='right',
            range=[min_price - padding, max_price + padding] 
        ),
        hovermode='x unified'
    )
    return fig, chart_df

def main():
    st.title("🤖 AI Market Tracker 2.0")
    
    if 'symbols' not in st.session_state:
        st.session_state['symbols'] = [{"symbol": "NVDA", "name": "NVIDIA Corporation"}, {"symbol": "AAPL", "name": "Apple Inc."}]
    if 'timeframes' not in st.session_state:
        st.session_state['timeframes'] = {}

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Settings & Controls")
        
        st.markdown("### 1. Add Company")
        search_query = st.text_input("Type a company or ticker:").strip()
        if st.button("➕ Add to Tracker"):
            if search_query:
                with st.spinner("Searching..."):
                    symbol, name = get_ticker_from_search(search_query)
                    if not any(d['symbol'] == symbol for d in st.session_state['symbols']):
                        st.session_state['symbols'].append({"symbol": symbol, "name": name})
                        st.success(f"Added {name} ({symbol})!")
                        st.rerun()
                    else:
                        st.warning("Already tracking this asset.")

        st.markdown("### 2. Manage Tracked Assets")
        tracked_names = [d['name'] for d in st.session_state['symbols']]
        to_remove = st.multiselect("Select to remove:", tracked_names)
        if st.button("🗑️ Remove Selected"):
            st.session_state['symbols'] = [d for d in st.session_state['symbols'] if d['name'] not in to_remove]
            st.rerun()

    # --- DASHBOARD ---
    if not st.session_state['symbols']:
        st.info("👈 Use the sidebar to search for a company and add it to your dashboard.")

    for asset in st.session_state['symbols']:
        symbol = asset['symbol']
        
        if symbol not in st.session_state['timeframes']:
            st.session_state['timeframes'][symbol] = "1M"

        df, news, info, real_name = fetch_market_data(symbol)
        
        if df is not None and not df.empty:
            st.markdown(f"---")
            
            # Header
            st.markdown(f"<h2>{real_name} <span style='color:gray; font-size: 0.6em;'>{symbol}</span></h2>", unsafe_allow_html=True)
            
            # Timeframe selector
            timeframe = st.radio(
                "Timeframe",
                ["1W", "1M", "3M", "6M", "YTD", "1Y"],
                horizontal=True,
                key=f"tf_{symbol}",
                label_visibility="collapsed"
            )
            
            # Chart & Metrics
            fig, chart_df = create_chart(df, timeframe)
            current_price = chart_df['Close'].iloc[-1]
            start_price = chart_df['Close'].iloc[0]
            price_change = current_price - start_price
            pct_change = (price_change / start_price) * 100
            
            color = "#34C759" if price_change >= 0 else "#FF3B30"
            sign = "+" if price_change >= 0 else ""
            st.markdown(f"<h1 style='margin-bottom: 0px;'>${current_price:,.2f}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h4 style='color: {color}; margin-top: 0px;'>{sign}{price_change:,.2f} ({sign}{pct_change:,.2f}%) <span style='color:gray; font-size:0.6em; font-weight:normal;'>Past {timeframe}</span></h4>", unsafe_allow_html=True)
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            # Stats row
            c1, c2, c3, c4 = st.columns(4)
            def format_large(num):
                if pd.isna(num) or num is None: return "-"
                if num >= 1e12: return f"{num/1e12:.3f}T"
                if num >= 1e9: return f"{num/1e9:.3f}B"
                if num >= 1e6: return f"{num/1e6:.3f}M"
                return f"{num:,.0f}"

            with c1: st.markdown(f"**Open:** ${chart_df['Open'].iloc[-1]:,.2f}  \n**High:** ${chart_df['High'].max():,.2f}  \n**Low:** ${chart_df['Low'].min():,.2f}")
            with c2: st.markdown(f"**Vol:** {format_large(chart_df['Volume'].iloc[-1])}  \n**Avg Vol:** {format_large(info.get('averageVolume'))}")
            with c3: st.markdown(f"**P/E:** {info.get('trailingPE', '-')}  \n**Mkt Cap:** {format_large(info.get('marketCap'))}")
            with c4: st.markdown(f"**52W H:** ${df['High'].max():,.2f}  \n**52W L:** ${df['Low'].min():,.2f}")
                
            st.markdown("<br>", unsafe_allow_html=True)

            # 1. Smart Algorithmic Prediction
            action, reason = get_smart_prediction(df)
            
            with st.expander(f"🤖 Smart Prediction for {symbol}", expanded=True):
                st.markdown(f"### Verdict: {action}")
                st.write(reason)
            
            # 2. Failsafe News Section
            with st.expander(f"📰 Recent News for {symbol}", expanded=False):
                for n in news:
                    title = n.get('title', 'Market Update')
                    publisher = n.get('publisher', 'Finance Network')
                    link = n.get('link', '#')
                    
                    st.markdown(f"**[{title}]({link})**")
                    st.markdown(f"<span style='color:gray; font-size: 0.8em;'>{publisher}</span>", unsafe_allow_html=True)
                    st.write("") 
                    
        else:
            st.warning(f"Could not load data for {symbol}. The ticker might be invalid.")

if __name__ == "__main__":
    main()
