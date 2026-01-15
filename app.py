import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas as pd
import pandas_ta as ta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="💰 AI Stock Auditor", layout="wide")

# --- TITLE & SIDEBAR ---
st.title("💰 AI Stock Forensic Auditor")
st.markdown("Enter a stock ticker to run the **7-Phase High-Profit Framework**.")

with st.sidebar:
    st.header("🔑 Configuration")
    # This input is safe because it's not saved anywhere
    api_key = st.text_input("Enter Gemini API Key", type="password")
    st.markdown("[Get Free Key Here](https://aistudio.google.com/)")
    st.info("💡 Note: For Indian stocks, add .NS (e.g. COALINDIA.NS)")

# --- DATA FETCHING FUNCTION ---
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Fetch 1 year of history for technicals
        hist = stock.history(period="1y")
        
        if hist.empty:
            return None
        
        # Calculate Technical Indicators
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        hist['SMA_50'] = ta.sma(hist['Close'], length=50)
        hist['SMA_200'] = ta.sma(hist['Close'], length=200)
        
        # Get latest values
        current_price = hist['Close'].iloc[-1]
        rsi_val = hist['RSI'].iloc[-1]
        sma_50_val = hist['SMA_50'].iloc[-1]
        sma_200_val = hist['SMA_200'].iloc[-1]
        
        # Get Fundamental Data
        info = stock.info
        
        # Helper to safely get data
        def safe_get(key):
            return info.get(key, "N/A")

        # Build the Data Dictionary
        data = {
            "Symbol": ticker,
            "Current Price": round(current_price, 2),
            "Market Cap": safe_get("marketCap"),
            "P/E Ratio": safe_get("trailingPE"),
            "PEG Ratio": safe_get("pegRatio"),
            "ROE": safe_get("returnOnEquity"),
            "Debt/Equity": safe_get("debtToEquity"),
            "Current Ratio": safe_get("currentRatio"),
            "Profit Growth": safe_get("earningsGrowth"),
            "Revenue Growth": safe_get("revenueGrowth"),
            "52W High": safe_get("fiftyTwoWeekHigh"),
            "RSI (14)": round(rsi_val, 2),
            "50 DMA": round(sma_50_val, 2),
            "200 DMA": round(sma_200_val, 2),
            "Tech Trend": "BULLISH" if current_price > sma_200_val else "BEARISH"
        }
        return data
    except Exception as e:
        return None

# --- MAIN APP LOGIC ---
ticker_input = st.text_input("Enter Ticker Symbol", value="COALINDIA.NS")
run_btn = st.button("🚀 Run Forensic Audit")

if run_btn:
    if not api_key:
        st.error("⚠️ Please enter your Gemini API Key in the sidebar first!")
    else:
        with st.spinner(f"🕵️‍♂️ Auditing {ticker_input}... Fetching live data & Analyzing..."):
            
            # 1. Configure AI
            genai.configure(api_key=api_key)
            
            # 2. Get Data
            stock_data = get_stock_data(ticker_input)
            
            if stock_data:
                # 3. Display Key Metrics Dashboard
                st.success("Data Fetched Successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"₹{stock_data['Current Price']}")
                col2.metric("RSI Momentum", stock_data['RSI (14)'])
                col3.metric("P/E Ratio", stock_data['P/E Ratio'])
                col4.metric("Trend (200 DMA)", stock_data['Tech Trend'])
                
                # 4. The AI Analysis Prompt
                model = genai.GenerativeModel('gemini-1.5-pro-latest')
                
                prompt = f"""
                You are a ruthless Hedge Fund Analyst. I have given you live data for {stock_data['Symbol']}.
                
                ### LIVE DATA:
                {stock_data}
                
                ### YOUR MISSION:
                Audit this stock against my strict "7-PHASE SAFETY & PROFIT FRAMEWORK".
                
                ### RULES:
                1. SAFETY: Debt/Equity < 1.0? Current Ratio > 1.5? (Crucial)
                2. EFFICIENCY: ROE > 15%?
                3. GROWTH: Revenue/Earnings Growth > 20%? (If < 10%, it's slow).
                4. VALUATION: PEG < 1.5? P/E vs Industry?
                5. TECHNICALS: Price > 200 DMA? RSI between 40-60 is entry, >75 is dangerous.
                
                ### OUTPUT FORMAT (Markdown):
                # 🚨 FINAL VERDICT: [BUY / SELL / WAIT]
                *(One sentence summary)*
                
                ## 🛡️ Phase 1: Safety & Solvency
                * [Pass/Fail] Analysis of Debt & Liquidity...
                
                ## 🚀 Phase 2: Growth Engine
                * [Pass/Fail] Analysis of Growth & ROE...
                
                ## 💰 Phase 3: Valuation Check
                * [Cheap/Expensive] Analysis of P/E and PEG...
                
                ## 📈 Phase 4: Technical Entry
                * **Trend:** (Above/Below 200 DMA)
                * **Momentum:** (RSI Analysis)
                * **Action:** (Buy Now / Wait for Dip to X)
                
                ## ⚠️ Key Risks
                (Bullet points)
                """
                
                # 5. Generate and Stream Result
                response = model.generate_content(prompt)
                st.markdown("---")
                st.markdown("## 📝 AI Forensic Report")
                st.markdown(response.text)
                
            else:
                st.error("❌ Error: Could not fetch data. Check the ticker symbol. For India, did you add '.NS'?")