import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Hedge Fund Terminal", layout="wide", page_icon="📈")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔑 Login")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("[Get Free Key](https://aistudio.google.com/)")
    st.divider()
    st.info("💡 Tip: Use .NS for India (e.g. RELIANCE.NS)")

# --- CACHED FUNCTIONS (Prevents Crashes) ---

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        if hist.empty: return None, None, None
        
        # Technical Indicators
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        hist['SMA_50'] = ta.sma(hist['Close'], length=50)
        hist['SMA_200'] = ta.sma(hist['Close'], length=200)
        
        info = stock.info
        
        def safe_get(key, unit=None):
            val = info.get(key, "N/A")
            if val == "N/A": return "N/A"
            if unit == "%" and isinstance(val, (int, float)): 
                return f"{val * 100:.2f}%" if val < 1 else f"{val:.2f}%" 
            return val

        data = {
            "Symbol": ticker,
            "Sector": info.get('sector', 'Unknown'),
            "Industry": info.get('industry', 'Unknown'),
            "Price": hist['Close'].iloc[-1],
            "50 DMA": hist['SMA_50'].iloc[-1],
            "200 DMA": hist['SMA_200'].iloc[-1],
            "RSI": hist['RSI'].iloc[-1],
            "Debt/Equity": info.get('debtToEquity', 'N/A'),
            "Current Ratio": info.get('currentRatio', 'N/A'),
            "ROE": info.get('returnOnEquity', 'N/A'),
            "Rev Growth": info.get('revenueGrowth', 'N/A'),
            "Earnings Growth": info.get('earningsGrowth', 'N/A'),
            "PE": info.get('trailingPE', 'N/A'),
            "PEG": info.get('pegRatio', 'N/A'),
            "Insider Hold": info.get('heldPercentInsiders', 'N/A'),
            "Inst Hold": info.get('heldPercentInstitutions', 'N/A')
        }
        
        return stock, hist, data
    except Exception:
        return None, None, None

@st.cache_data(ttl=3600)
def get_news_summary(_stock): # Underscore prevents hashing issues
    try:
        news = _stock.news[:3]
        return "\n".join([f"- {n['title']}" for n in news])
    except:
        return "No recent news."

def run_ai_analysis(api_key, prompt):
    genai.configure(api_key=api_key)
    # Simplified Fallback (Faster)
    models_to_try = ["gemini-1.5-flash", "gemini-1.5-pro"]
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception:
            continue
            
    return "❌ Error: AI models failed. Check Key/Quota."

# --- MAIN APP ---
st.title("📈 AI Hedge Fund Terminal")
st.caption("Institutional Grade Forensic Analysis • 7-Phase Framework")

# --- INPUT FORM (Prevents White Screen) ---
with st.form("analysis_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Enter Ticker Symbol", value="COALINDIA.NS")
    with col2:
        # The app will NOT run until this button is clicked
        submitted = st.form_submit_button("🚀 Run Analysis")

if submitted:
    if not api_key:
        st.error("Please enter API Key in sidebar.")
    else:
        with st.spinner(f"Analyzing {ticker}..."):
            stock, hist, data = get_stock_data(ticker)
            
            if stock and data:
                # --- TAB LAYOUT ---
                tab1, tab2, tab3 = st.tabs(["📝 Forensic Report", "📊 Charts", "📰 News"])
                
                with tab1:
                    news_text = get_news_summary(stock)
                    
                    prompt = f"""
                    Role: Senior Analyst. Audit {data['Symbol']} using the 7-Phase Framework.
                    
                    DATA: {data}
                    SECTOR: {data['Sector']}
                    NEWS: {news_text}
                    
                    TASK:
                    1. Interpret Safety (Debt/Equity, Current Ratio).
                    2. Interpret Growth (ROE, Revenue/Earnings).
                    3. Interpret Valuation (PE, PEG).
                    4. Sector Check (Mention key sector metrics).
                    5. Technicals (Price vs 200DMA, RSI).
                    6. Management (Insider/Inst Holding).
                    7. Risks.
                    
                    OUTPUT:
                    Provide a structured report with a FINAL VERDICT (BUY/SELL/WAIT) at the end.
                    """
                    
                    report = run_ai_analysis(api_key, prompt)
                    st.markdown(report)

                with tab2:
                    st.subheader("Price Action")
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=hist.index,
                                    open=hist['Open'], high=hist['High'],
                                    low=hist['Low'], close=hist['Close'], name='Price'))
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='orange'), name='50 DMA'))
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], line=dict(color='blue'), name='200 DMA'))
                    fig.update_layout(height=600, template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.subheader("Latest Headlines")
                    try:
                        for n in stock.news[:5]:
                            st.markdown(f"**{n['title']}**")
                            if 'link' in n:
                                st.markdown(f"[Read Article]({n['link']})")
                            st.divider()
                    except:
                        st.write("News unavailable.")
            else:
                st.error("Ticker not found. Try adding .NS")