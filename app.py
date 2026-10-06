import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# GEMINI IMPORT
# ============================================================

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Vendor Selection Assistant",
    page_icon="🏆",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

CRITERIA = [
    "Cost",
    "Quality",
    "Delivery",
    "Reliability",
    "Sustainability",
    "Payment Terms"
]

DEFAULT_WEIGHTS = {
    "Cost": 30,
    "Quality": 30,
    "Delivery": 15,
    "Reliability": 15,
    "Sustainability": 5,
    "Payment Terms": 5,
}

# Fixed Gemini model
GEMINI_MODEL = "gemini-3.8-flash"


# ============================================================
# SAMPLE DATASET
# ============================================================

SAMPLE = pd.DataFrame([
    ["Alpha Supplies", 8.0, 7.0, 9.0, 8.0, 6.0, 7.0],
    ["Bharat Components", 7.0, 9.0, 7.0, 9.0, 8.0, 8.0],
    ["CoreTech Industries", 9.0, 8.0, 8.0, 7.0, 7.0, 6.0],
    ["Delta Manufacturing", 6.0, 8.0, 6.0, 8.0, 9.0, 9.0],
    ["Elite Engineering", 8.0, 9.0, 8.0, 9.0, 8.0, 7.0],
    ["Fusion Suppliers", 9.0, 7.0, 8.0, 8.0, 6.0, 8.0],
    ["Global Parts Co.", 7.0, 8.0, 9.0, 7.0, 7.0, 7.0],
    ["Horizon Industrial", 8.0, 8.0, 7.0, 8.0, 9.0, 6.0],
    ["Indus Solutions", 6.0, 9.0, 7.0, 8.0, 8.0, 9.0],
    ["Jupiter Components", 9.0, 6.0, 9.0, 7.0, 6.0, 8.0],
], columns=["Vendor"] + CRITERIA)


# ============================================================
# DATA VALIDATION
# ============================================================

def validate(df):

    errors = []

    required = ["Vendor"] + CRITERIA

    # Check required columns
    missing_cols = [
        c for c in required
        if c not in df.columns
    ]

    if missing_cols:
        errors.append(
            f"Missing columns: {', '.join(missing_cols)}"
        )
        return errors

    # Check vendor names
    if df["Vendor"].isna().any():
        errors.append(
