import streamlit as st
import google.generativeai as genai
import yfinance as yf
import pandas_ta as ta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Hedge Fund Terminal", layout="centered", page_icon="🏦")

# --- CSS ---
st.markdown("""<style>.report-text { font-family: 'Courier New', monospace; }</style>""", unsafe_allow_html=True)

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🔑 Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("[Get Free Key](https://aistudio.google.com/)")
    
    st.divider()
    
    # --- MODEL SCANNER (The Fix) ---
    st.subheader("🧠 AI Brain")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Get all models that support generating content
            all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # Filter for just the 'Gemini' ones to keep it clean
            gemini_models = [m for m in all_models if 'gemini' in m]
            
            if not gemini_models:
                st.error("No Gemini models found for this key.")
                selected_model = "models/gemini-pro" # Fallback
            else:
                # Let user pick one from the list of ACTUAL available models
                selected_model = st.selectbox("Select Available Model", gemini_models, index=0)
                st.success(f"Connected: {selected_model}")
        except Exception as e:
            st.error(f"Connection Error: {e}")
            selected_model = "models/gemini-1.5-flash"
    else:
        selected_model = "models/gemini-1.5-flash"
        st.info("Enter Key to scan for models.")

    st.divider()
    st.info("💡 Tip: Use .NS for India (e.g. RELIANCE.NS)")

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: return None
        
        # Technicals
        hist['RSI'] = ta.rsi(hist['Close'], length=14)
        hist['SMA_200'] = ta.sma(hist['Close'], length=200)
        
        latest = hist.iloc[-1]
        info = stock.info
        
        # News
        try:
            news = "\n".join([f"- {n.get('title')}" for n in stock.news[:3]])
        except:
            news = "No news available."

        # Pure Data Payload
        return {
            "Symbol": ticker,
            "Price": f"{latest['Close']:.2f}",
            "RSI": f"{latest['RSI']:.2f}",
            "Trend": "Bullish" if latest['Close'] > latest['SMA_200'] else "Bearish",
            "PE": info.get('trailingPE', 'N/A'),
            "Debt_Equity": info.get('debtToEquity', 'N/A'),
            "ROE": info.get('returnOnEquity', 'N/A'),
            "News": news
        }
    except:
        return None

# --- AI ENGINE ---
def analyze_stock(api_key, model_name, data):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    prompt = f"""
    Act as a Senior Analyst. Write a 7-Phase Forensic Report for {data['Symbol']}.
    
    DATA:
    {data}
    
    INSTRUCTIONS:
    1. Safety: Analyze Debt/Equity.
    2. Growth: Analyze ROE.
    3. Valuation: Analyze P/E.
    4. Technicals: Current Price vs Trend, RSI status.
    5. Sentiment: Analyze the News headlines provided.
    
    FINAL OUTPUT:
    Give a strict "BUY/SELL/WAIT" verdict with 3 bullet points of rationale.
    """
    
    response = model.generate_content(prompt)
    return response.text

# --- MAIN UI ---
st.title("📝 AI Forensic Terminal")

with st.form("run_form"):
    ticker = st.text_input("Enter Ticker", value="COALINDIA.NS")
    submitted = st.form_submit_button("🚀 Run Analysis")

if submitted:
    if not api_key:
        st.error("Enter API Key first.")
    else:
        with st.spinner(f"Analyzing {ticker} using {selected_model}..."):
            data = get_market_data(ticker)
            if data:
                try:
                    report = analyze_stock(api_key, selected_model, data)
                    st.markdown("---")
                    st.markdown(report)
                except Exception as e:
                    st.error(f"AI Error: {e}")
            else:
                st.error("Ticker not found.")