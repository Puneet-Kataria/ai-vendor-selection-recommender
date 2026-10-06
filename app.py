import os
import io
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

st.set_page_config(page_title="AI Vendor Selection Assistant", page_icon="🏆", layout="wide")

CRITERIA = ["Cost", "Quality", "Delivery", "Reliability", "Sustainability", "Payment Terms"]
DEFAULT_WEIGHTS = {
    "Cost": 30,
    "Quality": 30,
    "Delivery": 15,
    "Reliability": 15,
    "Sustainability": 5,
    "Payment Terms": 5,
}

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


def validate(df):
    errors = []
    required = ["Vendor"] + CRITERIA
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {', '.join(missing_cols)}")
        return errors
    if df["Vendor"].isna().any() or (df["Vendor"].astype(str).str.strip() == "").any():
        errors.append("Vendor names cannot be blank.")
    if df["Vendor"].astype(str).duplicated().any():
        errors.append("Duplicate vendor names found. Each vendor must be unique.")
    for c in CRITERIA:
        numeric = pd.to_numeric(df[c], errors="coerce")
        if numeric.isna().any():
            errors.append(f"{c}: all values must be numeric and between 1 and 10.")
        elif ((numeric < 1) | (numeric > 10)).any():
            errors.append(f"{c}: scores must be between 1 and 10.")
    return errors


def score_vendors(df, weights):
    out = df.copy()
    for c in CRITERIA:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["Overall Score"] = sum(out[c] * (weights[c] / 100) for c in CRITERIA)
    out = out.sort_values("Overall Score", ascending=False).reset_index(drop=True)
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out


def get_ai_explanation(ranked, weights, model_name):
    api_key = st.session_state.get("api_key", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None, "No Gemini API key supplied. The analytical ranking is still available."
    if genai is None:
        return None, "The google-genai package is not installed. Run: pip install -r requirements.txt"

    top = ranked.iloc[0]
    runner = ranked.iloc[1] if len(ranked) > 1 else None
    records = ranked[["Rank", "Vendor", "Overall Score"] + CRITERIA].round(2).to_dict(orient="records")
    prompt = f"""
You are an AI procurement decision-support assistant.

Your task is to explain a vendor ranking produced by a deterministic weighted-scoring model.
Use ONLY the vendor data, scores, ranking and weights provided below. Do not invent facts,
financial details, certifications, locations, market reputation or other information.
Do not change any score or recalculate the ranking incorrectly.
Do not make an irreversible purchasing decision; provide decision support for a human procurement manager.

Scoring convention: every criterion is scored from 1 to 10, where a higher score means better performance.
For Cost, a higher score means lower/more competitive cost.

Weights: {weights}

Ranked vendors:
{records}

Produce a concise business recommendation with exactly these headings:
1. Recommendation
2. Why the leader ranks first
3. Key trade-offs
4. Risks / checks before final selection
5. Best alternative

The current leader is {top['Vendor']} with a score of {top['Overall Score']:.2f}.
The runner-up is {runner['Vendor']} with a score of {runner['Overall Score']:.2f} if one exists.
"""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=700,
                system_instruction=(
                    "You are a careful procurement analyst. Ground every statement in the supplied data. "
                    "If the data does not support a claim, say that it cannot be determined."
                ),
            ),
        )
        return response.text, None
    except Exception as exc:
        return None, f"Gemini request failed: {exc}"


st.title("🏆 AI Vendor Selection & Procurement Recommender")
st.caption("Data-driven vendor ranking with AI-assisted explanation and trade-off analysis")

with st.sidebar:
    st.header("1. Evaluation Weights")
    st.write("Set the relative importance of each criterion. Total must equal 100%.")
    weights = {}
    for c in CRITERIA:
        weights[c] = st.slider(c, 0, 100, DEFAULT_WEIGHTS[c], 5, key=f"w_{c}")
    total = sum(weights.values())
    if total != 100:
        st.error(f"Weights currently total {total}%. Adjust them to 100%.")
    else:
        st.success("Weights total 100%")

    st.divider()
    st.header("2. Gemini (Optional)")
    api_key_input = st.text_input("Gemini API key", type="password", help="Alternatively set GEMINI_API_KEY as an environment variable.")
    st.session_state.api_key = api_key_input
    model_name = st.text_input("Model", value="gemini-3.8-flash")
    st.caption("Keep API keys private. Do not paste confidential supplier information into a public demo app.")

st.subheader("3. Vendor Data")
col1, col2 = st.columns([1, 1])
with col1:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
with col2:
    if st.button("Load sample dataset", use_container_width=True):
        st.session_state.vendor_df = SAMPLE.copy()

if "vendor_df" not in st.session_state:
    st.session_state.vendor_df = SAMPLE.copy()

if uploaded is not None:
    try:
        st.session_state.vendor_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")

df = st.data_editor(st.session_state.vendor_df, num_rows="dynamic", use_container_width=True, key="vendor_editor")
st.session_state.vendor_df = df

with st.expander("Scoring guide"):
    st.markdown("- **1 = very weak performance** and **10 = excellent performance**.\n- For **Cost**, a higher score means a more competitive/lower cost.\n- For all criteria, higher is better.\n- The final score is the weighted sum of the six criterion scores.")

errors = validate(df)
if errors:
    st.error("Please fix the following data issues before analysis:")
    for e in errors:
        st.write(f"• {e}")
    st.stop()

if total != 100:
    st.warning("Set weights to exactly 100% to run the analysis.")
    st.stop()

ranked = score_vendors(df, weights)

st.subheader("4. Vendor Ranking")
leader = ranked.iloc[0]
metric_cols = st.columns(3)
metric_cols[0].metric("Recommended Vendor", leader["Vendor"])
metric_cols[1].metric("Overall Score", f"{leader['Overall Score']:.2f}/10")
metric_cols[2].metric("Vendors Evaluated", len(ranked))

st.dataframe(ranked[["Rank", "Vendor", "Overall Score"] + CRITERIA].style.format({"Overall Score": "{:.2f}"}), use_container_width=True, hide_index=True)

chart = px.bar(ranked, x="Vendor", y="Overall Score", title="Overall Vendor Score", text="Overall Score")
chart.update_traces(texttemplate="%{text:.2f}", textposition="outside")
chart.update_yaxes(range=[0, 10.8])
st.plotly_chart(chart, use_container_width=True)

st.subheader("5. What-if Analysis")
st.write("Change the weights in the sidebar and rerun the ranking. The order should change when priorities change.")

st.subheader("6. AI Procurement Insight")
if st.button("Generate AI recommendation", type="primary", use_container_width=True):
    with st.spinner("Generating grounded procurement analysis..."):
        explanation, error = get_ai_explanation(ranked, weights, model_name)
    if explanation:
        st.markdown(explanation)
    else:
        st.warning(error)
        st.info("The deterministic vendor ranking above remains valid even when the AI explanation is unavailable.")

st.divider()
st.caption("Decision-support prototype: final supplier selection should remain subject to human review, commercial negotiation, compliance checks and organizational procurement policy.")
