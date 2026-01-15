import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Hedge Fund Terminal", layout="wide", page_icon="📈")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔑 Login")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("[Get Free Key](https://aistudio.google.com/)")
    st.divider()
    st.info("💡 Tip: Use .NS for India (e.g. RELIANCE.NS)")

# --- CACHED FUNCTIONS (The Fix: Returns only serializable data) ---

@st.cache_data(ttl=3600)
def get_stock_analysis(ticker):
    """
    Fetches ALL data (Price, Techs, Fundamentals, News) in one go.
    Returns only DataFrames and Dictionaries (Safe for Caching).
    """
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Fetch History
        hist = stock.history(period="2y")
        if hist.empty: return None, None
        
        # 2. Calculate Technicals
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        hist['SMA_50'] = ta.sma(hist['Close'], length=50)
        hist['SMA_200'] = ta.sma(hist['Close'], length=200)
        
        # 3. Get Fundamentals
        info = stock.info
        
        # 4. Get News (Handled safely here)
        try:
            news_items = stock.news[:3]
            news_summary = "\n".join([f"- {n.get('title', 'No Title')}" for n in news_items])
        except:
            news_summary = "No recent news available."

        # 5. Package the Data
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
            "Inst Hold": info.get('heldPercentInstitutions', 'N/A'),
            "News_Summary": news_summary # Stored as string, safe for cache
        }
        
        return hist, data # Returns DataFrame and Dict (Safe!)
    except Exception as e:
        return None, None

def run_ai_analysis(api_key, prompt):
    genai.configure(api_key=api_key)
    # Smart Fallback List
    models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-latest", "gemini-pro"]
    
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except:
            continue
    return "❌ Error: All AI models failed. Please check your API Key."

# --- MAIN APP ---
st.title("📈 AI Hedge Fund Terminal")
st.caption("Institutional Grade Forensic Analysis • 7-Phase Framework")

# --- INPUT FORM ---
with st.form("analysis_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Enter Ticker Symbol", value="COALINDIA.NS")
    with col2:
        submitted = st.form_submit_button("🚀 Run Analysis")

if submitted:
    if not api_key:
        st.error("Please enter API Key in sidebar.")
    else:
        with st.spinner(f"Analyzing {ticker}..."):
            # Now returns only 2 items (hist, data)
            hist, data = get_stock_analysis(ticker)
            
            if data:
                # --- TAB LAYOUT ---
                tab1, tab2 = st.tabs(["📝 Forensic Report", "📊 Charts"])
                
                with tab1:
                    prompt = f"""
                    Role: Senior Analyst. Audit {data['Symbol']} using the 7-Phase Framework.
                    
                    DATA: {data}
                    SECTOR: {data['Sector']}
                    NEWS: {data['News_Summary']}
                    
                    TASK:
                    1. Safety (Debt/Equity, Current Ratio).
                    2. Growth (ROE, Revenue/Earnings).
                    3. Valuation (PE, PEG).
                    4. Sector Check (Mention key metrics for {data['Sector']}).
                    5. Technicals (Price vs 200DMA, RSI).
                    6. Management (Insider/Inst Holding).
                    7. Risks.
                    
                    OUTPUT:
                    Structured report with FINAL VERDICT (BUY/SELL/WAIT) at the end.
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
                    
                    st.divider()
                    st.subheader("Recent Headlines")
                    st.text(data['News_Summary'])

            else:
                st.error("Ticker not found or Data Unavailable. Try adding .NS")