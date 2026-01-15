import streamlit as st
import google.generativeai as genai
import yfinance as yf

st.set_page_config(page_title="System Check", layout="centered")
st.title("🛠️ Connection Debugger")

# 1. Inputs
api_key = st.text_input("Enter Gemini API Key", type="password")
ticker = st.text_input("Ticker", value="COALINDIA.NS")

if st.button("Run System Test"):
    if not api_key:
        st.error("⚠️ No API Key Entered")
    else:
        # TEST 1: Check Google AI Connection
        st.info("📡 1. Testing Connection to Google AI...")
        try:
            genai.configure(api_key=api_key)
            # Try the most stable model first
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content("Reply with 'OK' if you can hear me.")
            st.success(f"✅ AI Connected Successfully! Response: {response.text}")
        except Exception as e:
            st.error("❌ AI Connection FAILED")
            st.error(f"Error Details: {e}") # <--- THIS WILL SHOW THE REAL REASON
            st.stop()

        # TEST 2: Check Yahoo Finance Data
        st.info(f"📉 2. Fetching Data for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1mo")
            if hist.empty:
                st.error("❌ Ticker Not Found or No Data Returned.")
            else:
                price = hist['Close'].iloc[-1]
                st.success(f"✅ Data Fetched! Current Price: {price}")
                
                # TEST 3: Full Integration
                st.info("🧠 3. Running Analysis Test...")
                try:
                    prompt = f"The stock {ticker} is at {price}. Is this high or low?"
                    final_res = model.generate_content(prompt)
                    st.markdown("### ✅ Final Report:")
                    st.write(final_res.text)
                except Exception as e:
                    st.error(f"❌ Analysis Generation Failed: {e}")

        except Exception as e:
            st.error(f"❌ Data Fetch Error: {e}")