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
    """Generates realistic dummy data when Yahoo blocks the IP or Imports fail."""
    dates = pd.date_range(end=datetime.today(), periods=500)
    
    # Create a random walk for price
    base_price = 400.0
    returns = np.random.normal(0, 0.02, 500)
    price_path = base_price * (1 + returns).cumprod()
    
    hist = pd.DataFrame(index=dates)
    hist['Close'] = price_path
    hist['Open'] = price_path * (1 + np.random.normal(0, 0.005, 500))
    hist['High'] = hist[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, 500)))
    hist['Low'] = hist[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, 500)))
    hist['Volume'] = np.random.randint(100000, 5000000, 500)
    
    # Add Mock Technicals (Manual calculation to avoid import issues in mock)
    # Simple Moving Averages
    hist['SMA_50'] = hist['Close'].rolling(window=50).mean().fillna(base_price)
    hist['SMA_200'] = hist['Close'].rolling(window=200).mean().fillna(base_price)
    # Mock RSI (Simple random fluctuation around 50)
    hist['RSI'] = 50 + np.random.normal(0, 10, 500)
    
    # Mock Info Dict
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
        'heldPercentInstitutions': 0.35
    }
    
    return info, hist

# --- DATA ENGINE (CRASH PROOF) ---
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    is_live = True
    try:
        # LAZY IMPORT: We import here so the app loads even if this fails
        import yfinance as yf
        
        # 1. Try Live Fetch
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")
        
        if hist.empty:
            raise ValueError("Empty Data")
            
        info = stock.info
        if len(info) < 2: 
            raise ValueError("Blocked Info")
            
    except Exception:
        # 2. FALLBACK TO MOCK DATA AUTOMATICALLY
        is_live = False
        info, hist = generate_mock_data(ticker)

    # --- PROCESS DATA ---
    try:
        # Technicals (If live, calc them; if mock, they are already there)
        if is_live:
            # We use try/except specifically for pandas_ta to avoid crashing if missing
            try:
                import pandas_ta as ta
                hist['RSI'] = ta.rsi(hist['Close'], length=14)
                hist['SMA_50'] = ta.sma(hist['Close'], length=50)
                hist['SMA_200'] = ta.sma(hist['Close'], length=200)
            except ImportError:
                # Fallback if pandas_ta is missing
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
                hist['RSI'] = 50 # Default neutral
        
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

        metrics = {
            "Symbol": ticker,
            "Price": f"{latest['Close']:.2f}",
            "Market Cap": info.get('marketCap', 'N/A'),
            "D/E Ratio": safe_fmt(de_ratio),
            "Current Ratio": safe_fmt(info.get('currentRatio', None)),
            "ROE": safe_fmt(info.get('returnOnEquity', None), is_percent=True),
            "Rev Growth": safe_fmt(info.get('revenueGrowth', None), is_percent=True),
            "Profit Growth": safe_fmt(info.get('earningsGrowth', None), is_percent=True),
            "P/E": safe_fmt(info.get('trailingPE', None)),
            "PEG": safe_fmt(info.get('pegRatio', None)),
            "EV/EBITDA": safe_fmt(info.get('enterpriseToEbitda', None)),
            "RSI": f"{latest['RSI']:.2f}",
            "RS_Rating": rs_metric,
            "Trend": "UP 🟢" if latest['Close'] > latest['SMA_200'] else "DOWN 🔴",
            "Inst Hold": safe_fmt(info.get('heldPercentInstitutions', None), is_percent=True),
            "Sector_Info": sector_ctx,
            "News_Headlines": news_headlines,
            "Is_Live": is_live
        }
        return metrics, hist
    except Exception as e:
        return None, None

# --- AI ENGINE ---
def analyze_stock(api_key, model_name, data):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    val_focus = "EV/EBITDA (Cyclical)" if data['Sector_Info']['Is_Cyclical'] else "PEG Ratio (Growth)"
    
    prompt = f"""
    Act as a Senior Hedge Fund Analyst. Audit {data['Symbol']} using this 7-PHASE FRAMEWORK.
    DATA SOURCE: {'LIVE MARKET DATA' if data['Is_Live'] else 'SIMULATED SCENARIO (DEMO MODE)'}
    SECTOR CONTEXT: {data['Sector_Info']['Advice']}
    DATA: {data}
    
    FRAMEWORK:
    1. Safety: Debt/Equity {data['D/E Ratio']}, Current Ratio {data['Current Ratio']}.
    2. Profit: ROE {data['ROE']}, Growth {data['Profit Growth']}.
    3. Valuation: Focus on {val_focus}. P/E {data['P/E']}, PEG {data['PEG']}, EV/EBITDA {data['EV/EBITDA']}.
    4. Sector: Comment on sector metrics.
    5. Technicals: Trend {data['Trend']}, RSI {data['RSI']}.
    6. Management: Inst Hold {data['Inst Hold']}.
    7. Risks: List 2 key risks.
    
    OUTPUT:
    # 🔍 Analysis based on the 7-Phase Safety & Profit Framework
    # 🎯 VERDICT: [BUY / WATCH / SELL]
    **Reason:** (One sentence summary).
    (Continue with 7 numbered points)
    """
    response = model.generate_content(prompt)
    return response.text

# --- MAIN UI ---
st.title("📈 AI Hedge Fund Terminal")
st.caption("Institutional Grade Forensic Analysis • 7-Phase Framework")

with st.form("run_form"):
    ticker = st.text_input("Ticker Symbol", value="COALINDIA.NS")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        submitted = st.form_submit_button("🚀 Run Analysis", use_container_width=True)

if submitted:
    if not api_key:
        st.error("⚠️ Enter API Key")
    else:
        with st.spinner(f"Analyzing {ticker}..."):
            data, hist = get_market_data(ticker)
            
            if data and hist is not None:
                # Mode Banner
                if not data['Is_Live']:
                    st.warning("⚠️ MARKET DATA CONNECTION LIMITED: Switched to SIMULATION MODE for demonstration.")
                else:
                    st.success("✅ LIVE DATA CONNECTION ESTABLISHED")

                # DASHBOARD
                st.subheader(f"📊 {ticker} Dashboard")
                st.caption(f"Sector: {data['Sector_Info']['Sector']} | {data['Sector_Info']['Industry']}")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("EV / EBITDA", data['EV/EBITDA'])
                m2.metric("P/E Ratio", data['P/E'])
                m3.metric("Debt / Equity", data['D/E Ratio'])
                m4.metric("Current Ratio", data['Current Ratio'])
                
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("ROE", data['ROE'])
                t2.metric("PEG Ratio", data['PEG'])
                t3.metric("RSI (14)", data['RSI'])
                t4.metric("Trend", data['Trend'])
                
                st.divider()
                st.subheader("📉 Technical Breakout Check")
                st.plotly_chart(plot_technical_chart(hist, ticker), use_container_width=True)
                
                st.divider()
                st.subheader("📝 Forensic Analysis")
                try:
                    report = analyze_stock(api_key, selected_model, data)
                    st.markdown(report)
                except Exception as e:
                    st.error(f"AI Error: {e}")