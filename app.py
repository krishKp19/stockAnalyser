import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from datetime import datetime

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
    
    # Gemini API Only
    gemini_api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("[Get Free Gemini Key](https://aistudio.google.com/)")
    
    st.divider()
    
    fallback_models = ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "models/gemini-pro"]
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            gemini_models = [m for m in models if 'gemini' in m]
            if not gemini_models: gemini_models = fallback_models
        except:
            gemini_models = fallback_models
        selected_model = st.selectbox("AI Brain", gemini_models, index=0)
    else:
        st.selectbox("AI Brain", ["Enter Gemini Key First"], disabled=True)

    st.divider()
    st.info("💡 Tip: Use .NS for India (e.g. COALINDIA.NS)")
    
    st.markdown("---")
    st.markdown("<p class='version-text'>v6.2 | YahooQuery Engine (Keyless)</p>", unsafe_allow_html=True)

# --- HELPER: SECTOR CONTEXT ---
def get_sector_context(sector, industry):
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

    fig.add_trace(go.Candlestick(x=hist.index, open=hist['open'], high=hist['high'],
                                 low=hist['low'], close=hist['close'], name='Price'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='orange', width=1), name='50 DMA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], line=dict(color='blue', width=2), name='200 DMA'), row=1, col=1)

    colors = ['#00FF00' if row['open'] - row['close'] >= 0 else '#FF0000' for index, row in hist.iterrows()]
    fig.add_trace(go.Bar(x=hist.index, y=hist['volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark",
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", 
                      font=dict(color="white"), margin=dict(l=10, r=10, t=30, b=10))
    return fig

# --- SIGNAL CALCULATION LOGIC ---
def calculate_signals(vol_ratio, rev_growth, earn_growth, promoter_hold):
    vol_score = 0
    vol_msg = "Normal"
    if vol_ratio > 3.0: vol_score, vol_msg = 3, "Institutional Aggression"
    elif vol_ratio > 2.0: vol_score, vol_msg = 2, "Confirmed Breakout"
    elif vol_ratio > 1.2: vol_score, vol_msg = 1, "Rising Interest"
    
    oplev_score = 0
    oplev_msg = "No Leverage"
    oplev_ratio = 0.0
    if rev_growth > 0 and earn_growth > 0:
        oplev_ratio = earn_growth / rev_growth
        if oplev_ratio > 4.0: oplev_score, oplev_msg = 3, "Parabolic Economics"
        elif oplev_ratio > 2.0: oplev_score, oplev_msg = 2, "Strong Leverage"
        elif oplev_ratio > 1.0: oplev_score, oplev_msg = 1, "Healthy Scaling"
    
    prom_score = 0
    prom_msg = "Low Alignment"
    if promoter_hold > 0.6: prom_score, prom_msg = 3, "High Conviction"
    elif promoter_hold > 0.4: prom_score, prom_msg = 2, "Strong Skin-in-Game"
    elif promoter_hold > 0.2: prom_score, prom_msg = 1, "Moderate Confidence"

    final_score = (0.40 * oplev_score) + (0.35 * vol_score) + (0.25 * prom_score)
    
    verdict = "Ignore"
    if final_score > 2.4: verdict = "High Conviction Buy 🚀"
    elif final_score > 1.8: verdict = "Investigate 🔍"
    elif final_score > 1.0: verdict = "Watchlist 👀"

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

# --- DATA ENGINE (YAHOO QUERY) ---
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
        from yahooquery import Ticker
        stock = Ticker(ticker)
        
        # Helper to safely extract data from yahooquery's nested dicts
        def safe_get(module_data, key, default=None):
            if isinstance(module_data, dict) and ticker in module_data:
                if isinstance(module_data[ticker], dict):
                    return module_data[ticker].get(key, default)
            return default

        # 1. Fetch History
        hist = stock.history(period="2y")
        if isinstance(hist, dict) or hist.empty:
            st.error(f"❌ Failed to fetch price history for {ticker}. Check the symbol.")
            return None, None
            
        # YahooQuery returns a multi-index (symbol, date). We flatten it.
        if isinstance(hist.index, pd.MultiIndex):
            hist = hist.xs(ticker)

        # 2. Fetch Fundamentals
        profile = stock.asset_profile
        summary = stock.summary_detail
        fin_data = stock.financial_data
        key_stats = stock.key_stats

        if isinstance(profile, dict) and isinstance(profile.get(ticker), str):
            st.error(f"❌ API Error for {ticker}: {profile.get(ticker)}")
            return None, None

        # 3. Technicals
        hist['SMA_50'] = hist['close'].rolling(window=50).mean()
        hist['SMA_200'] = hist['close'].rolling(window=200).mean()
        
        # RSI
        delta = hist['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        hist['RSI'].fillna(50, inplace=True)

        # Volume Surge Data
        avg_vol_20 = hist['volume'].rolling(window=20).mean().iloc[-1]
        current_vol = hist['volume'].iloc[-1]
        vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 1.0

        # Calculations & Formatting
        sector = safe_get(profile, 'sector', 'Unknown')
        industry = safe_get(profile, 'industry', 'Unknown')
        sector_ctx = get_sector_context(sector, industry)

        rev_g = safe_get(fin_data, 'revenueGrowth', 0)
        prof_g = safe_get(fin_data, 'earningsGrowth', 0)
        prom_hold = safe_get(key_stats, 'heldPercentInsiders', 0)
        
        signals = calculate_signals(vol_ratio, rev_g, prof_g, prom_hold)

        cfo = safe_get(fin_data, 'operatingCashflow', None)
        ebitda = safe_get(fin_data, 'ebitda', None)
        cfo_to_ebitda = "N/A"
        if cfo and ebitda and ebitda != 0:
            cfo_to_ebitda = f"{(cfo / ebitda):.0%}"

        def safe_fmt(val, is_percent=False):
            if val is None or val == "N/A" or pd.isna(val): return "N/A"
            try:
                val = float(val)
                if is_percent: return f"{val * 100:.2f}%"
                return f"{val:.2f}"
            except:
                return str(val)

        def format_large_number(num):
            if num is None or pd.isna(num): return "N/A"
            try:
                num = float(num)
                if num > 1e9: return f"{num/1e9:.2f}B"
                if num > 1e6: return f"{num/1e6:.2f}M"
                return f"{num:.0f}"
            except: return str(num)
            
        de_ratio = safe_get(fin_data, 'debtToEquity', None)
        if de_ratio and de_ratio > 10: de_ratio = de_ratio / 100 # Adjust if given as %

        metrics = {
            "Symbol": ticker,
            "Price": f"{hist['close'].iloc[-1]:.2f}",
            "Market Cap": format_large_number(safe_get(summary, 'marketCap')),
            "D/E Ratio": safe_fmt(de_ratio),
            "Current Ratio": safe_fmt(safe_get(fin_data, 'currentRatio')),
            "ROE": safe_fmt(safe_get(fin_data, 'returnOnEquity'), is_percent=True),
            "Rev Growth": safe_fmt(rev_g, is_percent=True),
            "Profit Growth": safe_fmt(prof_g, is_percent=True),
            "P/E": safe_fmt(safe_get(summary, 'trailingPE')),
            "PEG": safe_fmt(safe_get(key_stats, 'pegRatio')),
            "EV/EBITDA": format_large_number(ebitda), # Display raw EBITDA if EV is complex
            "RSI": f"{hist['RSI'].iloc[-1]:.2f}",
            "RS_Rating": "N/A",
            "Trend": "UP 🟢" if hist['close'].iloc[-1] > hist['SMA_200'].iloc[-1] else "DOWN 🔴",
            "Inst Hold": safe_fmt(safe_get(key_stats, 'heldPercentInstitutions'), is_percent=True), 
            "Sector_Info": sector_ctx,
            "News_Headlines": ["Data fetched via YahooQuery backend."],
            "CFO": format_large_number(cfo),
            "EBITDA": format_large_number(ebitda),
            "CFO_to_EBITDA": cfo_to_ebitda,
            "Signals": signals,
            "Sales_TTM": format_large_number(safe_get(fin_data, 'totalRevenue')),
            "Sales_LastYr": "N/A" # Simplify to just TTM for yahooquery reliability
        }
        return metrics, hist
        
    except Exception as e:
        st.error(f"❌ YAHOOQUERY ERROR: {e}")
        return None, None

# --- AI ENGINE ---
def analyze_stock(gemini_key_param, model_name, data):
    genai.configure(api_key=gemini_key_param)
    model = genai.GenerativeModel(model_name)
    val_focus = "EV/EBITDA (Cyclical)" if data['Sector_Info']['Is_Cyclical'] else "PEG Ratio (Growth)"
    sig = data['Signals']
    
    prompt = f"""
    Act as a Senior Hedge Fund Analyst. Audit {data['Symbol']} using this 7-PHASE FRAMEWORK.
    DATA SOURCE: LIVE MARKET DATA
    SECTOR CONTEXT: {data['Sector_Info']['Advice']}
    
    ### 🚦 QUANT SIGNAL DIAGNOSTIC
    - **Stock Readiness Score:** {sig['Final_Score']:.2f} / 3.0 (Quantitative Verdict: {sig['Verdict']})
    - **Conflict Check:** {sig['Conflict']}
    - Volume Signal: {sig['Vol_Score']}/3 ({sig['Vol_Msg']})
    - Operating Leverage: {sig['OpLev_Score']}/3 ({sig['OpLev_Msg']})
    
    ### 📊 GROWTH CHECK
    - **Sales (TTM):** {data['Sales_TTM']}
    
    IMPORTANT: The "Readiness Score" is a strict mathematical baseline. 
    **YOUR JOB is to interpret it.** - If the Score is LOW ("Ignore") but Fundamentals (CFO, ROE, Sales Trend) are STRONG, override the signal and recommend "WATCH" or "BUY".
    - Explain WHY the score might be low (e.g., "Good fundamentals, but no volume momentum yet").
    
    DATA: {data}
    
    FRAMEWORK:
    1. Safety: Debt/Equity {data['D/E Ratio']}, Current Ratio {data['Current Ratio']}.
    2. Profit: ROE {data['ROE']}, Growth {data['Profit Growth']}. Op Leverage: {sig['OpLev_Ratio']:.2f}x.
    3. Valuation: Focus on {val_focus}. P/E {data['P/E']}, PEG {data['PEG']}, EV/EBITDA {data['EV/EBITDA']}.
    4. Sector: Comment on sector metrics.
    5. Technicals: Trend {data['Trend']}, RSI {data['RSI']}, Volume Surge {sig['Vol_Ratio']:.1f}x.
    6. Management: Evaluate overall stability based on metrics.
    7. Risks: List 2 key risks.
    
    OUTPUT:
    # 🔍 Analysis based on the 7-Phase Safety & Profit Framework
    # 🎯 VERDICT: [BUY / WATCH / SELL]
    **Thesis:** (Explain your verdict, referencing the Signal Score vs Fundamentals).
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
    if not gemini_api_key:
        st.error("⚠️ Please enter your Gemini API Key in the sidebar.")
    else:
        with st.spinner(f"Fetching backend data for {ticker}..."):
            data, hist = get_market_data(ticker)
            
            if data and hist is not None:
                st.success("✅ DATA CONNECTION ESTABLISHED")

                # DASHBOARD
                st.subheader(f"📊 {ticker} Dashboard")
                st.caption(f"Sector: {data['Sector_Info']['Sector']} | {data['Sector_Info']['Industry']}")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Market Cap", data['Market Cap'])
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
                f4.metric("Sales (Current)", data['Sales_TTM'])
                
                # --- SIGNAL BOARD (BOTTOM) ---
                st.divider()
                sig = data['Signals']
                st.subheader(f"🚦 Signal Radar | Score: {sig['Final_Score']:.2f}")
                
                s1, s2, s3 = st.columns(3)
                s1.metric("Volume Momentum", f"{sig['Vol_Score']}/3", sig['Vol_Msg'])
                s2.metric("Op. Leverage", f"{sig['OpLev_Score']}/3", sig['OpLev_Msg'])
                s3.metric("Promoter Confidence", f"{sig['Prom_Score']}/3", sig['Prom_Msg'])
                
                if sig['Conflict'] != "None":
                    st.info(f"💡 **Insight:** {sig['Conflict']}")

                st.divider()
                st.subheader("📉 Technical Breakout Check")
                st.plotly_chart(plot_technical_chart(hist, ticker), use_container_width=True)
                
                st.divider()
                st.subheader("📝 Forensic Analysis")
                try:
                    report = analyze_stock(gemini_api_key, selected_model, data)
                    st.markdown(report)
                except Exception as e:
                    st.error(f"AI Error: {e}")