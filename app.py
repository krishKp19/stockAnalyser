import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="💰 AI Hedge Fund Terminal", layout="wide", page_icon="📈")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔑 Login")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("[Get Free Key](https://aistudio.google.com/)")
    st.divider()
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker Symbol", value="COALINDIA.NS")
    st.info("💡 Tip: Use .NS for India (e.g. RELIANCE.NS)")

# --- HELPER FUNCTIONS ---

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
        
        # Safe Data Extraction
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
            "Book Value": info.get('bookValue', 'N/A'),
            "Insider Hold": info.get('heldPercentInsiders', 'N/A'),
            "Inst Hold": info.get('heldPercentInstitutions', 'N/A')
        }
        
        return stock, hist, data
    except Exception as e:
        return None, None, None

def get_news_summary(stock):
    try:
        news = stock.news[:3]
        return "\n".join([f"- {n['title']}" for n in news])
    except:
        return "No recent news."

def run_ai_analysis(api_key, prompt):
    """
    Universal Fallback System: Tries multiple model names until one works.
    """
    genai.configure(api_key=api_key)
    
    # List of models to try in order of preference
    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro",
        "gemini-1.0-pro"
    ]
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text # If successful, return text and exit loop
        except Exception:
            continue # If failed, try the next model in the list
            
    return "❌ Error: All AI models failed. Please check your API Key or Quota."

# --- MAIN APP ---
st.title("📈 AI Hedge Fund Terminal")
st.caption("Institutional Grade Forensic Analysis • 7-Phase Framework")

if st.button("🚀 Run Full Analysis"):
    if not api_key:
        st.error("Please enter API Key.")
    else:
        with st.spinner("🔄 Running 7-Phase Forensic Scan..."):
            stock, hist, data = get_stock_data(ticker)
            
            if stock and data:
                # --- TAB LAYOUT ---
                tab1, tab2, tab3 = st.tabs(["📝 Forensic Report", "📊 Charts", "📰 News"])
                
                with tab1:
                    # PREPARE PROMPT
                    news_text = get_news_summary(stock)
                    
                    prompt = f"""
                    You are a Senior Hedge Fund Analyst. Conduct a deep 7-PHASE FORENSIC AUDIT on {data['Symbol']}.
                    
                    ### THE DATA:
                    {data}
                    
                    ### SECTOR CONTEXT:
                    Sector: {data['Sector']} | Industry: {data['Industry']}
                    
                    ### RECENT NEWS:
                    {news_text}
                    
                    ### INSTRUCTIONS:
                    1. **Tone:** Professional, objective. NO "FAIL/PASS" labels. Use "Concerns", "Strong", "Neutral".
                    2. **Structure:** Follow the 7 Phases strictly.
                    3. **Format:** For each phase, give an **Interpretation** paragraph first, then a **Metrics Table** showing [Metric | Actual | Threshold/Goal].
                    4. **Verdict:** Place the Recommendation (BUY/SELL/WAIT) at the very END.

                    ### THE 7 PHASES TO COVER:
                    
                    **Phase 1: Safety & Solvency**
                    - Check Debt/Equity (Target < 1.0 or < 100%).
                    - Check Current Ratio (Target > 1.5).
                    
                    **Phase 2: The Profit Engine**
                    - Check ROE (Target > 15%).
                    - Check Growth (Target > 20% for high growth).
                    
                    **Phase 3: Valuation**
                    - Check P/E vs typical range.
                    - Check PEG (Target < 1.5).
                    
                    **Phase 4: Sector-Specific Logic**
                    - Based on the Sector ({data['Sector']}), mention specific metrics the user should manually check (e.g. NPA for Banks).
                    
                    **Phase 5: Technical Entry**
                    - Compare Price vs 200 DMA (Trend).
                    - Check RSI (40-60 is entry, >70 is hot).
                    
                    **Phase 6: Smart Money (Management)**
                    - Analyze Insider & Institutional Holding.
                    
                    **Phase 7: Exit Risks**
                    - What could go wrong? (Regulatory, commodity prices, governance).
                    
                    ### FINAL OUTPUT:
                    End with a clear section:
                    # 🎯 FINAL INVESTMENT VERDICT
                    **Recommendation:** [STRONG BUY / ACCUMULATE / WAIT / SELL]
                    **Summary:** (2 lines on why)
                    """
                    
                    # RUN AI WITH FALLBACK
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
                    for n in stock.news[:5]:
                        st.markdown(f"**{n['title']}**")
                        if 'link' in n:
                            st.markdown(f"[Read Article]({n['link']})")
                        st.divider()
            else:
                st.error("Ticker not found.")