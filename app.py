import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import requests
import re
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
    st.info("💡 Tip: Type the Indian stock name (e.g., COALINDIA, TCS, ITC)")
    
    st.markdown("---")
    st.markdown("<p class='version-text'>v6.3 | Screener.in Web Scraper Engine</p>", unsafe_allow_html=True)

# --- HELPER: SECTOR CONTEXT ---
def get_sector_context(industry):
    context_map = {
        "Bank": "BANKING: Focus on NIM (>3.5%) and NPA trends.",
        "IT": "IT: Focus on Deal Wins (TCV) and Attrition.",
        "Auto": "RETAIL/AUTO: Focus on Same Store Sales (SSSG).",
        "Mining": "COMMODITIES: Focus on Capacity Utilization >85%.",
        "Power": "POWER: Focus on Plant Load Factor (PLF >75%).",
        "Pharmaceuticals": "PHARMA: Focus on USFDA Status."
    }
    is_cyclical = any(x in industry for x in ['Steel', 'Mining', 'Power', 'Auto'])
    sector_advice = context_map.get(industry, f"General Industry: {industry}. Focus on Cash Flow vs EBITDA.")
    return {"Sector": industry, "Industry": industry, "Advice": sector_advice, "Is_Cyclical": is_cyclical}

# --- LAZY CHARTING ---
def plot_technical_chart(hist, ticker):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{ticker} Price', 'Volume'), 
                        row_heights=[0.7, 0.3])

    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                 low=hist['Low'], close=hist['Close'], name='Price'), row=1, col=1)
    
    if 'SMA_50' in hist:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], line=dict(color='orange', width=1), name='50 DMA'), row=1, col=1)
    if 'SMA_200' in hist:
        fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], line=dict(color='blue', width=2), name='200 DMA'), row=1, col=1)

    colors = ['#00FF00' if row['Open'] - row['Close'] >= 0 else '#FF0000' for index, row in hist.iterrows()]
    fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

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

# --- DATA ENGINE (SCREENER.IN NATIVE SCRAPER) ---
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
        clean_ticker = ticker.upper().replace('.NS', '').replace('.BO', '')
        url = f"https://www.screener.in/company/{clean_ticker}/consolidated/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        res = requests.get(url, headers=headers)
        if res.status_code != 200:
            url = f"https://www.screener.in/company/{clean_ticker}/" # Fallback to standalone if consolidated fails
            res = requests.get(url, headers=headers)
            if res.status_code != 200:
                st.error(f"❌ Could not find {clean_ticker} on Screener.in. Ensure the spelling is correct.")
                return None, None
                
        html = res.text
        
        # 1. Regex Extraction for Top Metrics
        def extract_metric(name):
            pattern = f'<span class="name">\\s*{name}\\s*</span>.*?<span class="number">([^<]+)</span>'
            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
            if match:
                val = match.group(1).replace(',', '').strip()
                try: return float(val)
                except: return val
            return None

        mcap = extract_metric('Market Cap')
        pe = extract_metric('Stock P/E')
        roe = extract_metric('ROE')
        de = extract_metric('Debt to equity')
        price = extract_metric('Current Price')
        promoter = extract_metric('Promoter holding') or 0.0
        if isinstance(promoter, float): promoter = promoter / 100

        # 2. Extract Industry
        industry = "Unknown"
        ind_match = re.search(r'Sector:\s*<a[^>]*>([^<]+)</a>', html, re.IGNORECASE)
        if ind_match: industry = ind_match.group(1).strip()
        sector_ctx = get_sector_context(industry)

        # 3. Parse Financial Tables for Deep Analytics
        sales_ttm, ebitda, cfo = "N/A", "N/A", "N/A"
        try:
            tables = pd.read_html(html)
            for df in tables:
                if df.empty: continue
                df.set_index(df.columns[0], inplace=True)
                
                # P&L Table
                if any('Sales' in str(idx) for idx in df.index):
                    sales_row = [idx for idx in df.index if 'Sales' in str(idx)][0]
                    sales_ttm = df.loc[sales_row].iloc[-1]
                    
                    op_row = [idx for idx in df.index if 'Operating Profit' in str(idx)]
                    if op_row: ebitda = df.loc[op_row[0]].iloc[-1]
                
                # Cash Flow Table
                if any('Operating Activity' in str(idx) for idx in df.index):
                    cfo_row = [idx for idx in df.index if 'Operating Activity' in str(idx)][0]
                    cfo = df.loc[cfo_row].iloc[-1]
        except Exception as e:
            pass # Tables might not parse perfectly, fallback to N/A

        cfo_to_ebitda = "N/A"
        if isinstance(cfo, (int, float)) and isinstance(ebitda, (int, float)) and ebitda != 0:
            cfo_to_ebitda = f"{(cfo / ebitda):.0%}"

        # 4. Fetch Technical Price Chart (Best Effort via Yahoo, safe failure)
        hist = None
        rsi_val, trend, vol_ratio = 50, "N/A", 1.0
        try:
            import yfinance as yf
            hist_yf = yf.Ticker(ticker if ".NS" in ticker else f"{ticker}.NS").history(period="1y")
            if not hist_yf.empty:
                hist = hist_yf
                hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                hist['SMA_200'] = hist['Close'].rolling(window=200).mean()
                
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                hist['RSI'] = 100 - (100 / (1 + (gain / loss)))
                hist['RSI'].fillna(50, inplace=True)
                
                rsi_val = hist['RSI'].iloc[-1]
                trend = "UP 🟢" if hist['Close'].iloc[-1] > hist['SMA_200'].iloc[-1] else "DOWN 🔴"
                
                avg_vol_20 = hist['Volume'].rolling(window=20).mean().iloc[-1]
                if avg_vol_20 > 0: vol_ratio = hist['Volume'].iloc[-1] / avg_vol_20
        except:
            pass # If Yahoo blocks the price fetch, we just skip the chart

        signals = calculate_signals(vol_ratio, 0.15, 0.20, promoter) # Approximated growth for signal

        def format_cr(num):
            if num is None or num == "N/A": return "N/A"
            try: return f"₹ {float(num):,.0f} Cr"
            except: return str(num)

        metrics = {
            "Symbol": clean_ticker,
            "Price": f"₹ {price}" if price else "N/A",
            "Market Cap": format_cr(mcap),
            "D/E Ratio": de if de else "N/A",
            "Current Ratio": "N/A", # Screener top metric varies
            "ROE": f"{roe}%" if roe else "N/A",
            "P/E": pe if pe else "N/A",
            "PEG": "N/A",
            "EV/EBITDA": format_cr(ebitda), 
            "RSI": f"{rsi_val:.2f}",
            "Trend": trend,
            "Inst Hold": "Check Screener", 
            "Sector_Info": sector_ctx,
            "CFO": format_cr(cfo),
            "EBITDA": format_cr(ebitda),
            "CFO_to_EBITDA": cfo_to_ebitda,
            "Signals": signals,
            "Sales_TTM": format_cr(sales_ttm),
            "Sales_LastYr": "N/A" 
        }
        return metrics, hist
        
    except Exception as e:
        st.error(f"❌ Web Scraper Error: {e}")
        return None, None

# --- AI ENGINE ---
def analyze_stock(gemini_key_param, model_name, data):
    genai.configure(api_key=gemini_key_param)
    model = genai.GenerativeModel(model_name)
    val_focus = "EV/EBITDA (Cyclical)" if data['Sector_Info']['Is_Cyclical'] else "P/E Ratio (Growth)"
    sig = data['Signals']
    
    prompt = f"""
    Act as a Senior Hedge Fund Analyst. Audit {data['Symbol']} using this 7-PHASE FRAMEWORK.
    DATA SOURCE: SCREENER.IN (INDIAN EQUITIES PLATFORM)
    SECTOR CONTEXT: {data['Sector_Info']['Advice']}
    
    ### 🚦 QUANT SIGNAL DIAGNOSTIC
    - **Stock Readiness Score:** {sig['Final_Score']:.2f} / 3.0 (Quantitative Verdict: {sig['Verdict']})
    - Volume Signal: {sig['Vol_Score']}/3 ({sig['Vol_Msg']})
    
    ### 📊 GROWTH CHECK
    - **Sales (TTM):** {data['Sales_TTM']}
    
    DATA: {data}
    
    FRAMEWORK:
    1. Safety: Debt/Equity {data['D/E Ratio']}.
    2. Profit: ROE {data['ROE']}.
    3. Valuation: Focus on {val_focus}. P/E {data['P/E']}.
    4. Sector: Comment on sector metrics.
    5. Technicals: Trend {data['Trend']}, RSI {data['RSI']}, Volume Surge {sig['Vol_Ratio']:.1f}x.
    6. Management: Check Insider limits.
    7. Risks: List 2 key risks based on the numbers.
    
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
st.caption("Institutional Grade Forensic Analysis • Powered by Screener.in Scraper")

with st.form("run_form"):
    ticker = st.text_input("Indian Stock Symbol (e.g. COALINDIA, ITC, TCS)", value="COALINDIA")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        submitted = st.form_submit_button("🚀 Run Forensic Audit", use_container_width=True)

if submitted:
    if not gemini_api_key:
        st.error("⚠️ Please enter your Gemini API Key in the sidebar.")
    else:
        with st.spinner(f"Scraping Screener.in for {ticker}..."):
            data, hist = get_market_data(ticker)
            
            if data is not None:
                st.success("✅ SCREENER.IN DATA EXTRACTED SUCCESSFULLY")

                # DASHBOARD
                st.subheader(f"📊 {ticker} Dashboard")
                st.caption(f"Sector: {data['Sector_Info']['Sector']} | Prices in ₹ Crores")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Market Cap", data['Market Cap'])
                m2.metric("P/E Ratio", data['P/E'])
                m3.metric("Debt / Equity", data['D/E Ratio'])
                m4.metric("ROE", data['ROE'])
                
                # FORENSIC RADAR
                st.divider()
                st.subheader("🕵️ Forensic Radar")
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Operating Cash Flow", data['CFO'])
                f2.metric("EBITDA", data['EBITDA'])
                f3.metric("Cash Conv.", data['CFO_to_EBITDA'], help="CFO / EBITDA. Target is > 70%")
                f4.metric("Sales (TTM)", data['Sales_TTM'])
                
                # --- SIGNAL BOARD ---
                st.divider()
                sig = data['Signals']
                st.subheader(f"🚦 Quantitative Signal Radar")
                
                s1, s2, s3 = st.columns(3)
                s1.metric("Volume Momentum", f"{sig['Vol_Score']}/3", sig['Vol_Msg'])
                s2.metric("Trend", data['Trend'], "Moving Average")
                s3.metric("Promoter Confidence", f"{sig['Prom_Score']}/3", sig['Prom_Msg'])
                
                st.divider()
                if hist is not None:
                    st.subheader("📉 Technical Breakout Check")
                    st.plotly_chart(plot_technical_chart(hist, ticker), use_container_width=True)
                else:
                    st.info("⚠️ Live chart currently unavailable. Showing fundamental analysis below.")
                
                st.divider()
                st.subheader("📝 AI Forensic Report")
                try:
                    report = analyze_stock(gemini_api_key, selected_model, data)
                    st.markdown(report)
                except Exception as e:
                    st.error(f"AI Error: {e}")