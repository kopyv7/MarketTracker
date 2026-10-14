import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure the Streamlit page layout and settings
st.set_page_config(
    page_title="Market Tracker & Buy Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def calculate_rsi(data, periods=14):
    """Calculates the Relative Strength Index (RSI) to determine if an asset is overbought or oversold."""
    close_delta = data['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if up.empty or down.empty:
        return pd.Series(dtype='float64')

    ma_up = up.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    ma_down = down.ewm(com=periods - 1, adjust=True, min_periods=periods).mean()
    
    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi

@st.cache_data(ttl=300) # Cache data for 5 minutes to prevent hitting API limits
def fetch_market_data(symbol, period="6mo"):
    """Fetches historical market data and calculates technical indicators."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            return None, [], symbol
            
        # Calculate 50-day Simple Moving Average
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Calculate 14-day RSI
        df['RSI'] = calculate_rsi(df)
        
        # Fetch recent news for context
        news = ticker.news[:3] if hasattr(ticker, 'news') else []
        
        # Try to fetch the name of the company/ETF automatically
        try:
            name = ticker.info.get('shortName', symbol)
        except:
            name = symbol
            
        return df, news, name
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None, [], symbol

def analyze_buy_signal(current_price, sma_50, rsi, week_high_52):
    """Analyzes the indicators to provide a buy recommendation and explain the market state."""
    reasons = []
    is_low = False
    signal = "HOLD / NEUTRAL"
    color = "normal"

    # RSI Logic (Below 30 is oversold/cheap, Above 70 is overbought/expensive)
    if rsi < 30:
        is_low = True
        reasons.append("RSI is below 30, indicating the asset is strongly oversold (cheap).")
    elif rsi < 45:
        reasons.append("RSI is on the lower end, showing cooling momentum.")
    elif rsi > 70:
        reasons.append("RSI is above 70, indicating the asset is overbought (expensive).")

    # Moving Average Logic
    if current_price < sma_50:
        is_low = True
        reasons.append(f"Price is trading below its 50-day average (${sma_50:.2f}), showing a short-term downtrend.")
    else:
        reasons.append("Price is trading above its 50-day average, showing short-term strength.")

    # Discount from 52-week high
    discount = ((week_high_52 - current_price) / week_high_52) * 100
    if discount > 10:
        is_low = True
        reasons.append(f"Trading at a {discount:.1f}% discount from its 52-week high.")

    # Determine final signal
    if is_low and rsi < 40:
        signal = "STRONG BUY (Market is Low) 🟢"
        color = "inverse"
    elif is_low:
        signal = "ACCUMULATE (Good Entry Point) 🟡"
    elif rsi > 70:
        signal = "OVEREXTENDED (Wait for Dip) 🔴"
        
    return signal, color, reasons, is_low

def main():
    st.title("📈 Real-Time ETF & Stock Market Tracker")
    st.markdown("Track your portfolio, analyze if the market is low, and understand the 'why' behind price movements.")
    
    # Initialize session state to store our dynamic list of tickers
    if 'symbols' not in st.session_state:
        st.session_state['symbols'] = ["ZSP.TO", "XEQT.TO", "VDY.TO", "GC=F"]
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Dashboard Controls")
        
        st.markdown("### Add New Asset")
        new_symbol = st.text_input("Enter Ticker Symbol (e.g. AAPL, NVDA, TSLA):").strip().upper()
        if st.button("➕ Add Symbol"):
            if new_symbol and new_symbol not in st.session_state['symbols']:
                st.session_state['symbols'].append(new_symbol)
                st.rerun()
            elif new_symbol in st.session_state['symbols']:
                st.warning(f"{new_symbol} is already being tracked!")
                
        st.markdown("### Manage Assets")
        symbols_to_remove = st.multiselect("Select symbols to remove:", st.session_state['symbols'])
        if st.button("🗑️ Remove Selected"):
            for sym in symbols_to_remove:
                st.session_state['symbols'].remove(sym)
            if symbols_to_remove:
                st.rerun()
                
        st.markdown("---")
        if st.button("🔄 Refresh Data Now"):
            st.cache_data.clear()
            st.success("Data refreshed!")
            
        st.markdown("---")
        st.markdown("""
        **How it works:**
        - **RSI (Relative Strength Index):** Measures momentum. < 30 means it's heavily sold off (good time to buy).
        - **50-Day SMA:** Average price over 50 days. Price below this often indicates a dip.
        """)

    for symbol in st.session_state['symbols']:
        
        # Fetch data
        with st.spinner(f"Fetching data for {symbol}..."):
            result = fetch_market_data(symbol)
            if result and len(result) == 3:
                df, news, asset_name = result
            else:
                df, news, asset_name = None, [], symbol
            
        if df is not None and not df.empty:
            st.markdown(f"### {asset_name} (`{symbol}`)")
            
            # Extract latest data points
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change = current_price - prev_price
            pct_change = (price_change / prev_price) * 100
            
            latest_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
            sma_50 = df['SMA_50'].iloc[-1]
            high_52 = df['High'].max()
            
            # Analyze
            signal, delta_color, reasons, is_low = analyze_buy_signal(current_price, sma_50, latest_rsi, high_52)
            
            # Layout metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2f} ({pct_change:.2f}%)")
            with col2:
                st.metric("14-Day RSI", f"{latest_rsi:.1f}", "Oversold (<30) / Overbought (>70)", delta_color="off")
            with col3:
                st.metric("50-Day Avg", f"${sma_50:.2f}")
            with col4:
                st.metric("Recommendation", signal)
                
            # Explanation / "Why is it low?" Section
            with st.expander(f"🧠 Analysis & Why is {symbol} moving?"):
                st.markdown("#### Technical Breakdown:")
                for reason in reasons:
                    st.markdown(f"- {reason}")
                    
                st.markdown("#### Market Context / Recent Headlines:")
                if news:
                    for article in news:
                        title = article.get('title', 'No title')
                        publisher = article.get('publisher', 'Unknown source')
                        link = article.get('link', '#')
                        st.markdown(f"- [{title}]({link}) *(Source: {publisher})*")
                else:
                    st.info("No recent news headlines found for this ticker right now. The market movement is likely tracking general macroeconomic trends (e.g., interest rates, inflation data).")
            
            # Chart
            st.line_chart(df['Close'], height=250, use_container_width=True)
            st.markdown("---")
        else:
            st.warning(f"Could not load data for {symbol}. Market might be closed or symbol is invalid.")

if __name__ == "__main__":
    main()