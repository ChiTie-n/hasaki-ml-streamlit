"""
Main entry point - Auto redirect to Welcome
"""
import streamlit as st

st.set_page_config(
    page_title="Pricing Decision Support",
    page_icon="💰",
    layout="wide"
)

# Auto navigate to Welcome page
st.switch_page("pages/00_Welcome.py")
