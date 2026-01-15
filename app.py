import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas_ta as ta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Hedge Fund Terminal (Lite)", layout="centered", page_icon="📝")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .report-text { font-family: 'Courier New', monospace; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("🔑 Login")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("[Get Free Key](https://aistudio.google.com/)")
    st.divider()
    st.info("💡 Tip: Use .NS for India (e.g. RELIANCE.NS)")

# --- CACHED DATA ENGINE (Optimized for Stability) ---
@st.cache_data(ttl=3600)
def get_fundamental_data(ticker):
    """
    Fetches only the scalars needed for the report. 
    No heavy DataFrames are returned to the UI.
    """
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Get Technicals (Calc internal, return scalars)
        hist = stock.history(period="2y")
        if hist.empty: return None
        
        # Calc Indicators using pandas_ta
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        hist['SMA_50'] = ta.sma(hist['Close'], length=50)
        hist['SMA_200'] = ta.sma(hist['Close'], length=200)
        
        # Extract latest values
        latest = hist.iloc[-1]
        
        # 2. Get Fundamentals
        info = stock.info
        
        # 3. Get News Summary (Text only)
        try:
            news = stock.news[:3]
            news_text = "\n".join([f"- {n.get('title')}" for n in news])
        except:
            news_text = "News Unavailable"

        # 4. Build the "Prompt Payload" (Pure Dictionary)
        data = {
            "Symbol": ticker,
            "Sector": info.get('sector', 'Unknown'),
            "Industry": info.get('industry', 'Unknown'),
            "Price": f"{latest['Close']:.2f}",
            "RSI": f"{latest['RSI']:.2f}",
            "50DMA": f"{latest['SMA_50']:.2f}",
            "200DMA": f"{latest['SMA_200']:.2f}",
            "Tech_Trend": "BULLISH" if latest['Close'] > latest['SMA_200'] else "BEARISH",
            
            # Fundamentals
            "Debt_Equity": info.get('debtToEquity', 'N/A'),
            "Current_Ratio": info.get('currentRatio', 'N/A'),
            "ROE": info.get('returnOnEquity', 'N/A'),
            "Rev_Growth": info.get('revenueGrowth', 'N/A'),
            "Earnings_Growth": info.get('earningsGrowth', 'N/A'),
            "PE": info.get('trailingPE', 'N/A'),
            "PEG": info.get('pegRatio', 'N/A'),
            "Insider_Hold": info.get('heldPercentInsiders', 'N/A'),
            "Inst_Hold": info.get('heldPercentInstitutions', 'N/A'),
            
            # News
            "Recent_News": news_text
        }
        return data
    except Exception as e:
        return None

def run_forensic_ai(api_key, data):
    genai.configure(api_key=api_key)
    # Priority: Flash (Fast) -> Pro (Deep) -> Standard
    models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    prompt = f"""
    Role: Institutional Market Analyst.
    Task: Write a "7-Phase Forensic Investment Report" for {data['Symbol']}.
    
    ### MARKET DATA:
    {data}
    
    ### INSTRUCTIONS:
    Analyze the data above and produce a structured report. 
    - DO NOT use "Pass/Fail" labels. Use professional terms like "Concerning," "Robust," "Elevated."
    - For every phase, provide a **Data Table** comparing Actual vs Ideal.
    
    ### REPORT STRUCTURE:
    
    **Phase 1: Solvency & Safety**
    - Analyze Debt/Equity (Ideal < 100%) and Current Ratio (Ideal > 1.5).
    - *Verdict on Bankruptcy Risk.*
    
    **Phase 2: Growth & Efficiency**
    - Analyze ROE (Ideal > 15%) and Growth Rates.
    - *Verdict on Quality.*
    
    **Phase 3: Valuation**
    - Analyze P/E and PEG. Is it cheap or a trap?
    
    **Phase 4: Sector Nuances**
    - Mention specific metrics relevant to {data['Sector']} that the user should check manually.
    
    **Phase 5: Technical Setup**
    - Current Price vs 200 DMA ({data['Tech_Trend']}).
    - RSI Status ({data['RSI']}).
    - *Actionable Signal: Buy / Wait / Sell.*
    
    **Phase 6: Management & Sponsorship**
    - Comment on Insider/Institutional holding.
    
    **Phase 7: Risk Factors**
    - List 3 key risks (Regulatory, Macro, Company-specific).
    
    ### FINAL VERDICT:
    # 🎯 INVESTMENT RECOMMENDATION: [BUY / ACCUMULATE / HOLD / SELL]
    **Summary:** (One concise paragraph justifying the decision).
    """
    
    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except:
            continue
    return "❌ Error: AI Service Unavailable. Please check API Key."

# --- MAIN UI ---
st.title("📝 AI Forensic Terminal (v1)")
st.caption("Pure Metrics • No Charts • Maximum Stability")

# Use a Form to prevent auto-reloading crashes
with st.form("analysis_form"):
    ticker = st.text_input("Enter Ticker", value="COALINDIA.NS")
    submitted = st.form_submit_button("🚀 Generate Forensic Report")

if submitted:
    if not api_key:
        st.error("⚠️ Please enter your Gemini API Key in the sidebar.")
    else:
        with st.spinner(f"Running Forensic Audit on {ticker}..."):
            # 1. Get Data (Lightweight)
            data = get_fundamental_data(ticker)
            
            if data:
                # 2. Run AI
                report = run_forensic_ai(api_key, data)
                
                # 3. Display Report
                st.markdown("---")
                st.markdown(report)
                st.markdown("---")
                st.caption("Data Source: Yahoo Finance | Analysis: Google Gemini")
            else:
                st.error("❌ Ticker Not Found or Data Error.")