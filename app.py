import streamlit as st

st.set_page_config(title="Adult Income Prediction Dashboard", page_icon="📈", layout="wide")

# Immediately redirect to the EDA page so the root page doesn't appear as a separate page
try:
	st.switch_page("pages/1_📊_EDA.py")
except Exception:
	# Fallback hint if switch_page isn't available
	st.write("Redirecting to EDA... Use the sidebar if not redirected.")


