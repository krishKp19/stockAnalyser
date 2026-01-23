import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Hedge Fund Terminal", layout="wide", page_icon="Hz")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    [data-testid="stMetricValue"] { font-size: 24px; color: #ffffff; }
    [data-testid="stMetricLabel"] { font-size: 14px; color: #888888; }
    .stAlert { background-color: #1e1e1e; color: #ff4b4b; border: 1px solid #ff4b4b; }
    .version-text { font-size: 12px; color: #444; text-align: center; margin-top: 50px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔑 Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("[Get Free Key](https://aistudio.google.com/)")
    st.divider()
    
    fallback_models = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            gemini_models = [m for m in models if 'gemini' in m]
            if not gemini_models: gemini_models = fallback_models
        except:
            gemini_models = fallback_models
        selected_model = st.selectbox("AI Brain", gemini_models, index=0)
    else:
        st.selectbox("AI Brain", ["Enter Key First"], disabled=True)

    st.divider()
    st.info("💡 Tip: Use .NS for India (e.g. RELIANCE.NS)")
    
    st.markdown("---")
    st.markdown("<p class='version-text'>v5.1 | Neutral Forensic Data</p>", unsafe_allow_html=True)

# --- HELPER: SECTOR CONTEXT ---
def get_sector_context(info):
    sector = info.get('sector', 'Unknown')
    industry = info.get('industry', 'Unknown')
    context_map = {
        "Financial Services": "BANKING: Focus on NIM (>3.5%) and NPA trends.",
        "Technology": "IT: Focus on Deal Wins (TCV) and Attrition.",
        "Consumer Cyclical": "RETAIL/AUTO: Focus on Same Store Sales (SSSG).",
        "Basic Materials": "COMMODITIES: Focus on Capacity Utilization >85%.",
        "Utilities": "POWER: Focus on Plant Load Factor (PLF >75%).",
        "Healthcare": "PHARMA: Focus on USFDA Status.",
        "Energy": "OIL/GAS: Watch Crude Oil prices."
    }
    is_cyclical = any(x in sector for x in ['Basic Materials', 'Energy', 'Utilities', 'Industrials'])
    sector_advice = context_map.get(sector, f"General Sector: {sector}. Focus on Cash Flow.")
    return {"Sector": sector, "Industry": industry, "Advice": sector_advice, "Is_Cyclical": is_cyclical}

# --- HELPER: LAZY CHARTING ---
def plot_technical_chart(hist, ticker):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'Volume'), 
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                 low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='orange', width=1), name='50 DMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], line=dict(color='blue', width=2), name='200 DMA'), row=1, col=1)

    colors = ['#00FF00' if row['Open'] - row['Close'] >= 0 else '#FF0000' for index, row in hist.iterrows()]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark",
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", 
                      font=dict(color="white"), margin=dict(l=10, r=10, t=30, b=10))
    return fig

# --- MOCK DATA GENERATOR ---
def generate_mock_data(ticker):
    """Generates realistic dummy data when Yahoo blocks the IP."""
    dates = pd.date_range(end=datetime.today(), periods=500)
    base_price = 400.0
    returns = np.random.normal(0, 0.02, 500)
    price_path = base_price * (1 + returns).cumprod()
    
    hist = pd.DataFrame(index=dates)
    hist['Close'] = price_path
    hist['Open'] = price_path * (1 + np.random.normal(0, 0.005, 500))
    hist['High'] = hist[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, 500)))
    hist['Low'] = hist[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, 500)))
    hist['Volume'] = np.random.randint(100000, 5000000, 500)
    
    # Mock Technicals
    hist['SMA_50'] = hist['Close'].rolling(window=50).mean().fillna(base_price)
    hist['SMA_200'] = hist['Close'].rolling(window=200).mean().fillna(base_price)
    hist['RSI'] = 50 + np.random.normal(0, 10, 500)
    
    info = {
        'sector': 'Basic Materials (Simulated)',
        'industry': 'Other Industrial Metals',
        'marketCap': 50000000000,
        'debtToEquity': 85.5,
        'currentRatio': 1.8,
        'returnOnEquity': 0.18,
        'revenueGrowth': 0.12,
        'earningsGrowth': 0.15,
        'trailingPE': 12.5,
        'pegRatio': 0.9,
        'enterpriseToEbitda': 6.5,
        'heldPercentInstitutions': 0.35,
        'operatingCashflow': 8000000000,
        'ebitda': 10000000000
    }
    
    return info, hist

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    is_live = True
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        if hist.empty: raise ValueError("Empty Data")
        info = stock.info
        if len(info) < 2: raise ValueError("Blocked Info")
    except Exception:
        is_live = False
        info, hist = generate_mock_data(ticker)

    try:
        # Technicals
        if is_live:
            try:
                import pandas_ta as ta
                hist['RSI'] = ta.rsi(hist['Close'], length=14)
                hist['SMA_50'] = ta.sma(hist['Close'], length=50)
                hist['SMA_200'] = ta.sma(hist['Close'], length=200)
            except ImportError:
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
                hist['RSI'] = 50 
        
        latest = hist.iloc[-1]
        
        # Benchmark RS
        rs_metric = "N/A (Mode: Sim)"
        if is_live:
            try:
                import yfinance as yf
                benchmark_symbol = "^NSEI" if ".NS" in ticker else "^GSPC"
                bench = yf.Ticker(benchmark_symbol)
                bench_hist = bench.history(start=hist.index[0], end=hist.index[-1])
                if len(hist) > 126 and len(bench_hist) > 126:
                    stock_6m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-126]) - 1
                    bench_6m = (bench_hist['Close'].iloc[-1] / bench_hist['Close'].iloc[-126]) - 1
                    rs_value = (stock_6m - bench_6m) * 100
                    rs_metric = f"{rs_value:+.2f}%"
            except: pass

        # News
        news_headlines = ["Live news unavailable in Simulation Mode."]
        if is_live:
            try:
                import yfinance as yf
                stock_for_news = yf.Ticker(ticker)
                news = stock_for_news.news
                news_headlines = [n['title'] for n in news[:5]] if news else ["No recent news."]
            except: pass

        sector_ctx = get_sector_context(info)
        
        def safe_fmt(val, is_percent=False):
            if val is None or val == "N/A": return "N/A"
            if isinstance(val, (int, float)):
                if is_percent: return f"{val * 100:.2f}%"
                return f"{val:.2f}"
            return val
            
        de_ratio = info.get('debtToEquity', None)
        if de_ratio and de_ratio > 10: de_ratio = de_ratio / 100

        # v5.1 Forensic Data (Neutral)
        cfo = info.get('operatingCashflow', None)
        ebitda = info.get('ebitda', None)
        cfo_to_ebitda = "N/A"
        
        if cfo and ebitda and ebitda != 0:
            ratio = cfo / ebitda
            cfo_to_ebitda = f"{ratio:.0%}"

        metrics = {
            "Symbol": ticker,
            "Price": f"{latest['Close']:.2f}",
            "Market Cap": info.get('marketCap', 'N/A'),
            "D/E Ratio": safe_fmt(de_ratio),
            "Current Ratio": safe_fmt(info.get('currentRatio', None)),
            "ROE": safe_fmt(info.get('returnOnEquity', None), is_percent=True),
            "Rev Growth": safe_fmt(info.get('revenueGrowth', None), is_percent=True),
            "Profit Growth": safe_fmt(info.get('earningsGrowth', None), is_percent=True),
            "P/E":