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
    /* Clean up the top padding */
    .block-container {
        padding-top: 2rem;
    }
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #ffffff;
    }
    [data-testid="stMetricLabel"] {
        font-size: 14px;
        color: #888888;
    }
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
        
        # 1. Fetch History
        hist = stock.history(period="2y")
        if hist.empty: return None
        
        # 2. Benchmark for Relative Strength (RS)
        benchmark_symbol = "^NSEI" if ".NS" in ticker else "^GSPC"
        try:
            bench = yf.Ticker(benchmark_symbol)
            bench_hist = bench.history(start=hist.index[0], end=hist.index[-1])
            
            if len(hist) > 126 and len(bench_hist) > 126:
                stock_6m_ret = (hist['Close'].iloc[-1] / hist['Close'].iloc[-126]) - 1
                bench_6m_ret = (bench_hist['Close'].iloc[-1] / bench_hist['Close'].iloc[-126]) - 1
                rs_value = (stock_6m_ret - bench_6m_ret) * 100
                rs_metric = f"{rs_value:+.2f}%" # FIX: Removed "vs Market" to prevent spillover
            else:
                rs_metric = "N/A"
        except:
            rs_metric = "N/A"

        # 3. Technical Indicators
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        hist['SMA_50'] = ta.sma(hist['Close'], length=50)
        hist['SMA_200'] = ta.sma(hist['Close'], length=200)
        
        latest = hist.iloc[-1]
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
            
            # Fundamentals
            "D/E Ratio": safe_fmt(info.get('debtToEquity', None), is_percent=False),
            "Current Ratio": safe_fmt(info.get('currentRatio', None)),
            "ROE": safe_fmt(info.get('returnOnEquity', None), is_percent=True),
            "Rev Growth": safe_fmt(info.get('revenueGrowth', None), is_percent=True),
            "Profit Growth": safe_fmt(info.get('earningsGrowth', None), is_percent=True),
            "P/E": safe_fmt(info.get('trailingPE', None)),
            "PEG": safe_fmt(info.get('pegRatio', None)),
            "EV/EBITDA": safe_fmt(info.get('enterpriseToEbitda', None)),
            
            # Technicals
            "RSI": f"{latest['RSI']:.2f}",
            "RS_Rating": rs_metric,
            "50 DMA": f"{latest['SMA_50']:.2f}",
            "200 DMA": f"{latest['SMA_200']:.2f}",
            "Trend": "UP 🟢" if latest['Close'] > latest['SMA_200'] else "DOWN 🔴",
            
            "Inst Hold": safe_fmt(info.get('heldPercentInstitutions', None), is_percent=True),
        }
        
        # D/E Logic Fix
        if metrics["D/E Ratio"] != "N/A" and float(metrics["D/E Ratio"]) > 10: 
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
    - EV/EBITDA: {data['EV/EBITDA']} (Target < 10 for Mfg).
    - P/E: {data['P/E']} vs PEG: {data['PEG']}.
    - *Is it Overvalued?*
    
    **Phase 4: Sector Check**
    - Identify sector based on ticker. Mention 1 specific metric relevant to this sector (e.g. NIM for Banks, Same Store Sales for Retail) user should check.
    
    **Phase 5: Technical Entry**
    - Trend: {data['Trend']} (Price vs 200 DMA).
    - Relative Strength (RS): {data['RS_Rating']} (Positive = Leader).
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

# FIX: Center Button Layout
with st.form("run_form"):
    ticker = st.text_input("Ticker Symbol", value="COALINDIA.NS")
    
    # Create 3 columns, put button in the middle one to center it
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        submitted = st.form_submit_button("🚀 Run Analysis", use_container_width=True)

if submitted:
    if not api_key:
        st.error("⚠️ Enter API Key in Sidebar")
    else:
        with st.spinner(f"Fetching Data & Analyzing {ticker}..."):
            data = get_market_data(ticker)
            
            if data:
                # --- SECTION 1: BOLD METRICS DASHBOARD ---
                st.subheader("📊 Key Metrics")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("EV / EBITDA", data['EV/EBITDA'], help="Target < 10")
                m2.metric("P/E Ratio", data['P/E'])
                m3.metric("Debt / Equity", data['D/E Ratio'], help="Target < 1.0")
                m4.metric("Current Ratio", data['Current Ratio'], help="Target > 1.5")
                
                st.divider()
                
                t1, t2, t3, t4 = st.columns(4)
                t1.metric("ROE", data['ROE'], help="Target > 15%")
                # FIX: Renamed label to include context, kept value short
                t2.metric("Rel. Strength (vs Mkt)", data['RS_Rating'], help="vs Benchmark (6 Months)")
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