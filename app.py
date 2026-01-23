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
    .signal-badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }
    .signal-3 { background-color: #00FF00; color: black; } /* Strong */
    .signal-2 { background-color: #90EE90; color: black; } /* Good */
    .signal-1 { background-color: #FFFF00; color: black; } /* Weak */
    .signal-0 { background-color: #FF6347; color: white; } /* None */
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
    st.markdown("<p class='version-text'>v5.2 | Signal Engine Integration</p>", unsafe_allow_html=True)

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
        'earningsGrowth': 0.30, # High growth for leverage check
        'trailingPE': 12.5,
        'pegRatio': 0.9,
        'enterpriseToEbitda': 6.5,
        'heldPercentInstitutions': 0.35,
        'heldPercentInsiders': 0.45,
        'operatingCashflow': 8000000000,
        'ebitda': 10000000000
    }
    
    return info, hist

# --- SIGNAL CALCULATION LOGIC (LAYER 1 & 2) ---
def calculate_signals(vol_ratio, rev_growth, earn_growth, promoter_hold):
    # 1. Volume Signal
    vol_score = 0
    vol_msg = "Normal"
    if vol_ratio > 3.0: vol_score, vol_msg = 3, "Institutional Aggression"
    elif vol_ratio > 2.0: vol_score, vol_msg = 2, "Confirmed Breakout"
    elif vol_ratio > 1.2: vol_score, vol_msg = 1, "Rising Interest"
    
    # 2. Operating Leverage Signal
    oplev_score = 0
    oplev_msg = "No Leverage"
    oplev_ratio = 0.0
    
    if rev_growth > 0 and earn_growth > 0:
        oplev_ratio = earn_growth / rev_growth
        if oplev_ratio > 4.0: oplev_score, oplev_msg = 3, "Parabolic Economics"
        elif oplev_ratio > 2.0: oplev_score, oplev_msg = 2, "Strong Leverage"
        elif oplev_ratio > 1.0: oplev_score, oplev_msg = 1, "Healthy Scaling"
    
    # 3. Promoter Holding Signal (Governance)
    # Note: Using raw decimal from API (0.45 = 45%)
    prom_score = 0
    prom_msg = "Low Alignment"
    if promoter_hold > 0.6: prom_score, prom_msg = 3, "High Conviction"
    elif promoter_hold > 0.4: prom_score, prom_msg = 2, "Strong Skin-in-Game"
    elif promoter_hold > 0.2: prom_score, prom_msg = 1, "Moderate Confidence"

    # 4. Composite Score (Layer 3)
    # Weight: 40% OpLev + 35% Vol + 25% Promo
    final_score = (0.40 * oplev_score) + (0.35 * vol_score) + (0.25 * prom_score)
    
    # 5. Interpretation (Layer 4)
    verdict = "Ignore"
    if final_score > 2.4: verdict = "High Conviction Buy 🚀"
    elif final_score > 1.8: verdict = "Investigate 🔍"
    elif final_score > 1.0: verdict = "Watchlist 👀"

    # 6. Conflict Detection (Optional)
    conflict_msg = "None"
    if vol_score >= 2 and oplev_score == 0:
        conflict_msg = "Speculative Spike (Price moving without fundamentals)"
    elif oplev_score >= 2 and vol_score == 0:
        conflict_msg = "Early Fundamental Story (Good numbers, market hasn't noticed)"

    return {
        "Vol_Score": vol_score, "Vol_Msg": vol_msg, "Vol_Ratio": vol_ratio,
        "OpLev_Score": oplev_score, "OpLev_Msg": oplev_msg, "OpLev_Ratio": oplev_ratio,
        "Prom_Score": prom_score, "Prom_Msg": prom_msg,
        "Final_Score": final_score, "Verdict": verdict, "Conflict": conflict_msg
    }

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
        avg_vol_20 = 0
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
        
        # Calculate Volume Surge Data
        avg_vol_20 = hist['Volume'].rolling(window=20).mean().iloc[-1]
        current_vol = hist['Volume'].iloc[-1]
        vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

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

        # Data for Signals
        rev_g = info.get('revenueGrowth', 0)
        prof_g = info.get('earningsGrowth', 0)
        prom_hold = info.get('heldPercentInsiders', 0)
        
        # Calculate Signals
        signals = calculate_signals(vol_ratio, rev_g, prof_g, prom_hold)

        # Forensic: Cash Flow
        cfo = info.get('operatingCashflow', None)
        ebitda = info.get('ebitda', None)
        cfo_to_ebitda = "N/A"
        if cfo and ebitda and ebitda != 0:
            cfo_to_ebitda = f"{(cfo / ebitda):.0%}"

        metrics = {
            "Symbol": ticker,
            "Price": f"{latest['Close']:.2f}",
            "Market Cap": info.get('marketCap', 'N/A'),
            "D/E Ratio": safe_fmt(de_ratio),
            "Current Ratio": safe_fmt(info.get('currentRatio', None)),
            "ROE": safe_fmt(info.get('returnOnEquity', None), is_percent=True),
            "Rev Growth": safe_fmt(rev_g, is_percent=True),
            "Profit Growth": safe_fmt(prof_g, is_percent=True),
            "P/E": safe_fmt(info.get('trailingPE', None)),
            "PEG": safe_fmt(info.get('pegRatio', None)),
            "EV/EBITDA": safe_fmt(info.get('enterpriseToEbitda', None)),
            "RSI": f"{latest['RSI']:.2f}",
            "RS_Rating": rs_metric,
            "Trend": "UP 🟢" if latest['Close'] > latest['SMA_200'] else "DOWN 🔴",
            "Inst Hold": safe_fmt(info.get('heldPercentInstitutions', None), is_percent=True),
            "Sector_Info": sector_ctx,
            "News_Headlines": news_headlines,
            "Is_Live": is_live,
            "CFO": safe_fmt(cfo),
            "EBITDA": safe_fmt(ebitda),
            "CFO_to_EBITDA": cfo_to_ebitda,
            # Signal Data
            "Signals": signals
        }
        return metrics, hist
    except Exception as e:
        return None, None

# --- AI ENGINE ---
def analyze_stock(api_key, model_name, data):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    val_focus = "EV/EBITDA (Cyclical)" if data['Sector_Info']['Is_Cyclical'] else "PEG Ratio (Growth)"
    sig = data['Signals']
    
    prompt = f"""
    Act as a Senior Hedge Fund Analyst. Audit {data['Symbol']} using this 7-PHASE FRAMEWORK.
    DATA SOURCE: {'LIVE MARKET DATA' if data['Is_Live'] else 'SIMULATED SCENARIO (DEMO MODE)'}
    SECTOR CONTEXT: {data['Sector_Info']['Advice']}
    
    ### 🚦 SIGNAL DIAGNOSTIC (The Core Thesis)
    - **Stock Readiness Score:** {sig['Final_Score']:.2f} / 3.0 ({sig['Verdict']})
    - **Conflict Check:** {sig['Conflict']}
    - Volume Signal: {sig['Vol_Score']}/3 ({sig['Vol_Msg']})
    - Operating Leverage: {sig['OpLev_Score']}/3 ({sig['OpLev_Msg']})
    - Promoter Confidence: {sig['Prom_Score']}/3 ({sig['Prom_Msg']})
    
    DATA: {data}
    
    FRAMEWORK:
    1. Safety: Debt/Equity {data['D/E Ratio']}, Current Ratio {data['Current Ratio']}.
    2. Profit: ROE {data['ROE']}, Growth {data['Profit Growth']}. Op Leverage: {sig['OpLev_Ratio']:.2f}x.
    3. Valuation: Focus on {val_focus}. P/E {data['P/E']}, PEG {data['PEG']}, EV/EBITDA {data['EV/EBITDA']}.
    4. Sector: Comment on sector metrics.
    5. Technicals: Trend {data['Trend']}, RSI {data['RSI']}, Volume Surge {sig['Vol_Ratio']:.1f}x.
    6. Management: Inst Hold {data['Inst Hold']}, Promoter Hold {sig['Prom_Score']} rating.
    7. Risks: List 2 key risks.
    
    OUTPUT:
    # 🔍 Analysis: {sig['Verdict']}
    
    **Thesis:** (Explain the Readiness Score and Conflict Check in 2 sentences).
    
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
                if not data['Is_Live']:
                    st.warning("⚠️ MARKET DATA CONNECTION LIMITED: Switched to SIMULATION MODE for demonstration.")
                else:
                    st.success("✅ LIVE DATA CONNECTION ESTABLISHED")

                # --- SIGNAL BOARD (NEW) ---
                sig = data['Signals']
                st.subheader(f"🚦 Signal Radar | Score: {sig['Final_Score']:.2f}")
                
                s1, s2, s3 = st.columns(3)
                s1.metric("Volume Momentum", f"{sig['Vol_Score']}/3", sig['Vol_Msg'])
                s2.metric("Op. Leverage", f"{sig['OpLev_Score']}/3", sig['OpLev_Msg'])
                s3.metric("Promoter Confidence", f"{sig['Prom_Score']}/3", sig['Prom_Msg'])
                
                if sig['Conflict'] != "None":
                    st.info(f"💡 **Insight:** {sig['Conflict']}")

                st.divider()

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
                
                # FORENSIC RADAR
                st.divider()
                st.subheader("🕵️ Forensic Radar")
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Operating Cash Flow", data['CFO'])
                f2.metric("EBITDA", data['EBITDA'])
                f3.metric("Cash Conv.", data['CFO_to_EBITDA'])
                f4.metric("Inst. Hold", data['Inst Hold'])
                
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