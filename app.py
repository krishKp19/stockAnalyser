import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas_ta as ta
import pandas as pd

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Hedge Fund Terminal", layout="wide", page_icon="🏦")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .metric-label { font-size: 14px; color: #888; margin-bottom: 5px; }
    .metric-value { font-size: 24px; font-weight: bold; color: #fff; }
    .metric-trend-up { color: #00FF00; font-size: 14px; }
    .metric-trend-down { color: #FF0000; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔑 Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("[Get Free Key](https://aistudio.google.com/)")
    
    st.divider()
    
    # Model Selector
    if api_key:
        try:
            genai.configure(api_key=api_key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            gemini_models = [m for m in models if 'gemini' in m]
            selected_model = st.selectbox("AI Brain", gemini_models, index=0)
        except:
            selected_model = "models/gemini-1.5-flash"
    else:
        selected_model = "models/gemini-1.5-flash"

    st.divider()
    st.info("💡 Tip: Use .NS for India (e.g. RELIANCE.NS)")

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Fetch History (2 Years for 200 DMA)
        hist = stock.history(period="2y")
        if hist.empty: return None
        
        # 2. Benchmark for Relative Strength (RS)
        # If Indian stock, compare with Nifty 50 (^NSEI), else S&P 500 (^GSPC)
        benchmark_symbol = "^NSEI" if ".NS" in ticker else "^GSPC"
        try:
            bench = yf.Ticker(benchmark_symbol)
            bench_hist = bench.history(start=hist.index[0], end=hist.index[-1])
            
            # Calculate 6-Month Return (approx 126 trading days)
            if len(hist) > 126 and len(bench_hist) > 126:
                stock_6m_ret = (hist['Close'].iloc[-1] / hist['Close'].iloc[-126]) - 1
                bench_6m_ret = (bench_hist['Close'].iloc[-1] / bench_hist['Close'].iloc[-126]) - 1
                rs_value = (stock_6m_ret - bench_6m_ret) * 100 # Outperformance %
                rs_metric = f"{rs_value:+.2f}% vs Market"
            else:
                rs_metric = "N/A (New Listing)"
        except:
            rs_metric = "N/A"

        # 3. Technical Indicators
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        hist['SMA_50'] = ta.sma(hist['Close'], length=50)
        hist['SMA_200'] = ta.sma(hist['Close'], length=200)
        
        latest = hist.iloc[-1]
        
        # 4. Fundamental Metrics (From Info)
        info = stock.info
        
        def safe_fmt(val, is_percent=False):
            if val is None or val == "N/A": return "N/A"
            if isinstance(val, (int, float)):
                if is_percent: return f"{val * 100:.2f}%"
                return f"{val:.2f}"
            return val

        # Metric Dictionary
        metrics = {
            "Symbol": ticker,
            "Price": f"{latest['Close']:.2f}",
            "Market Cap": info.get('marketCap', 'N/A'),
            
            # Phase 1: Safety
            "D/E Ratio": safe_fmt(info.get('debtToEquity', None), is_percent=False), # yf returns this as ratio often, check logic
            "Current Ratio": safe_fmt(info.get('currentRatio', None)),
            "Pledged %": "Check Manually", # yf often misses this for India
            
            # Phase 2: Profit
            "ROE": safe_fmt(info.get('returnOnEquity', None), is_percent=True),
            "Rev Growth": safe_fmt(info.get('revenueGrowth', None), is_percent=True),
            "Profit Growth": safe_fmt(info.get('earningsGrowth', None), is_percent=True),
            "Margins": safe_fmt(info.get('operatingMargins', None), is_percent=True),
            
            # Phase 3: Valuation
            "P/E": safe_fmt(info.get('trailingPE', None)),
            "PEG": safe_fmt(info.get('pegRatio', None)),
            "EV/EBITDA": safe_fmt(info.get('enterpriseToEbitda', None)),
            
            # Phase 4: Technicals
            "RSI": f"{latest['RSI']:.2f}",
            "RS_Rating": rs_metric,
            "50 DMA": f"{latest['SMA_50']:.2f}",
            "200 DMA": f"{latest['SMA_200']:.2f}",
            "Trend": "UP 🟢" if latest['Close'] > latest['SMA_200'] else "DOWN 🔴",
            
            # Phase 6: Management
            "Inst Hold": safe_fmt(info.get('heldPercentInstitutions', None), is_percent=True),
            "Promoter Hold": safe_fmt(info.get('heldPercentInsiders', None), is_percent=True)
        }
        
        # Correction: YF usually returns debtToEquity as a whole number (e.g. 120 for 1.2), checking logic
        if metrics["D/E Ratio"] != "N/A" and float(metrics["D/E Ratio"]) > 10: 
             # If it's > 10, it's likely percentage (e.g. 54.3%), convert to ratio 0.54
             metrics["D/E Ratio"] = f"{float(metrics['D/E Ratio']) / 100:.2f}"

        return metrics
    except Exception as e:
        return None

# --- AI ENGINE ---
def analyze_stock(api_key, model_name, data):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    prompt = f"""
    Act as a Hedge Fund Manager. Audit {data['Symbol']} using this strict 7-PHASE FRAMEWORK.
    
    ### DATA INPUTS:
    {data}
    
    ### FRAMEWORK INSTRUCTIONS:
    
    **Phase 1: Safety Filter**
    - Debt/Equity: {data['D/E Ratio']} (Target < 1.0).
    - Current Ratio: {data['Current Ratio']} (Target > 1.5).
    - *Verdict on Financial Health.*
    
    **Phase 2: Profit Engine**
    - ROE: {data['ROE']} (Target > 15%).
    - Growth: Rev {data['Rev Growth']} | Profit {data['Profit Growth']}.
    - *Verdict on Quality.*
    
    **Phase 3: Valuation (The Price)**
    - EV/EBITDA: {data['EV/EBITDA']} (Target < 10 for Mfg, higher for others).
    - P/E: {data['P/E']} vs PEG: {data['PEG']}.
    - *Is it Overvalued?*
    
    **Phase 4: Sector Check**
    - Identify the sector based on the ticker. Mention 1 specific metric relevant to this sector (e.g. NIM for Banks, Same Store Sales for Retail) that user should check.
    
    **Phase 5: Technical Entry**
    - Trend: {data['Trend']} (Price vs 200 DMA).
    - Relative Strength (RS): {data['RS_Rating']} (Positive = Leader, Negative = Laggard).
    - RSI: {data['RSI']} (40-60 Entry Zone).
    - *Timing Verdict.*
    
    **Phase 6: Smart Money**
    - Inst Holding: {data['Inst Hold']}. High is good.
    
    **Phase 7: Exit Risks**
    - List 2 major risks.
    
    ### FINAL OUTPUT:
    # 🎯 INVESTMENT VERDICT: [BUY / ACCUMULATE / WATCH / SELL]
    **Reason:** (One sentence summary).
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- MAIN UI ---
st.title("📈 AI Hedge Fund Terminal (v2.0)")

# Form to prevent reloading
with st.form("run_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Ticker Symbol", value="COALINDIA.NS")
    with col2:
        submitted = st.form_submit_button("🚀 Run Analysis")

if submitted:
    if not api_key:
        st.error("⚠️ Enter API Key in Sidebar")
    else:
        with st.spinner(f"Fetching Data & Analyzing {ticker}..."):
            data = get_market_data(ticker)
            
            if data:
                # --- SECTION 1: BOLD METRICS DASHBOARD ---
                st.subheader("📊 Key Metrics")
                
                # Row 1: Valuation & Safety
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("EV / EBITDA", data['EV/EBITDA'], help="Target < 10 (Indus/Mfg)")
                m2.metric("P/E Ratio", data['P/E'])
                m3.metric("Debt / Equity", data['D/E Ratio'], help="Target < 1.0")
                m4.metric("Current Ratio", data['Current Ratio'], help="Target > 1.5")
                
                st.divider()
                
                # Row 2: Growth & Technicals
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("ROE", data['ROE'], help="Target > 15%")
                t2.metric("Relative Strength (6M)", data['RS_Rating'], help="vs Benchmark")
                t3.metric("RSI (14)", data['RSI'], help="30-70 Range")
                t4.metric("Trend (200 DMA)", data['Trend'])
                
                # --- SECTION 2: AI REPORT ---
                st.divider()
                st.subheader("📝 Forensic Analysis")
                try:
                    report = analyze_stock(api_key, selected_model, data)
                    st.markdown(report)
                except Exception as e:
                    st.error(f"AI Error: {e}")
            else:
                st.error("❌ Ticker Not Found or Data Unavailable")