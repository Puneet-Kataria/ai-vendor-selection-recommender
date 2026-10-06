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
    missing_cols = [c for c in required if c not in df.columns]

    if missing_cols:
        errors.append(
            f"Missing columns: {', '.join(missing_cols)}"
        )
        return errors

    # Check vendor names
    if df["Vendor"].isna().any():
        errors.append("Vendor names cannot be blank.")

    if (df["Vendor"].astype(str).str.strip() == "").any():
        errors.append("Vendor names cannot be blank.")

    # Check duplicate vendor names
    vendor_names = df["Vendor"].astype(str).str.strip().str.lower()

    if vendor_names.duplicated().any():
        errors.append(
            "Duplicate vendor names found. Each vendor must be unique."
        )

    # Check criterion values
    for c in CRITERIA:

        numeric = pd.to_numeric(
            df[c],
            errors="coerce"
        )

        if numeric.isna().any():
            errors.append(
                f"{c}: all values must be numeric and between 1 and 10."
            )

        elif ((numeric < 1) | (numeric > 10)).any():
            errors.append(
                f"{c}: scores must be between 1 and 10."
            )

    return errors


# ============================================================
# VENDOR SCORING
# ============================================================

def score_vendors(df, weights):

    out = df.copy()

    # Convert all criteria to numeric
    for c in CRITERIA:
        out[c] = pd.to_numeric(
            out[c],
            errors="coerce"
        )

    # Calculate weighted score
    out["Overall Score"] = sum(
        out[c] * (weights[c] / 100)
        for c in CRITERIA
    )

    # Rank vendors
    out = out.sort_values(
        "Overall Score",
        ascending=False
    ).reset_index(drop=True)

    out.insert(
        0,
        "Rank",
        range(1, len(out) + 1)
    )

    return out


# ============================================================
# GEMINI AI ANALYSIS
# ============================================================

def get_ai_explanation(ranked, weights, model_name):

    # --------------------------------------------------------
    # Read API key securely from Streamlit Secrets
    # --------------------------------------------------------

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = ""

    if not api_key:
        return (
            None,
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY under "
            "Streamlit → Settings → Secrets."
        )

    # --------------------------------------------------------
    # Check whether google-genai is installed
    # --------------------------------------------------------

    if genai is None or types is None:
        return (
            None,
            "The google-genai package is not installed. "
            "Add google-genai to requirements.txt and redeploy."
        )

    # --------------------------------------------------------
    # Identify leader and runner-up
    # --------------------------------------------------------

    top = ranked.iloc[0]

    runner = (
        ranked.iloc[1]
        if len(ranked) > 1
        else None
    )

    # --------------------------------------------------------
    # Prepare vendor information for Gemini
    # --------------------------------------------------------

    records = (
        ranked[
            ["Rank", "Vendor", "Overall Score"] + CRITERIA
        ]
        .round(2)
        .to_dict(orient="records")
    )

    # --------------------------------------------------------
    # Construct grounded prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI procurement decision-support assistant.

Your task is to explain a vendor ranking produced by a
deterministic weighted-scoring model.

IMPORTANT RULES:

1. Use ONLY the vendor data, scores, rankings and weights
   provided below.

2. Do NOT invent facts about vendors.

3. Do NOT invent financial information, certifications,
   locations, reputation, capacity, delivery history or
   any other external information.

4. Do NOT change the calculated ranking.

5. Do NOT independently calculate a different ranking.

6. Do NOT make an irreversible purchasing decision.

7. Provide decision support for a human procurement manager.

8. Clearly identify limitations where the available data
   does not support a conclusion.

SCORING CONVENTION:

Every criterion is scored from 1 to 10.

Higher scores indicate better performance.

For Cost, a higher score means a more competitive/lower cost.

CURRENT WEIGHTS:

{weights}

RANKED VENDORS:

{records}

CURRENT LEADER:

{top['Vendor']} with an overall score of
{top['Overall Score']:.2f}/10

RUNNER-UP:

{
    runner['Vendor']
    if runner is not None
    else "No runner-up available"
}

with a score of

{
    f"{runner['Overall Score']:.2f}/10"
    if runner is not None
    else "N/A"
}

Provide a concise business recommendation using EXACTLY
these headings:

### 1. Recommendation

### 2. Why the leader ranks first

### 3. Key trade-offs

### 4. Risks / checks before final selection

### 5. Best alternative

Keep the analysis practical and concise.

Do not make claims that cannot be supported by the
provided vendor data.
"""

    # --------------------------------------------------------
    # Call Gemini
    # --------------------------------------------------------

    try:

        client = genai.Client(
            api_key=api_key
        )

        response = client.models.generate_content(

            model=model_name,

            contents=prompt,

            config=types.GenerateContentConfig(

                temperature=0.2,

                max_output_tokens=700,

                system_instruction=(
                    "You are a careful procurement analyst. "
                    "Ground every statement in the supplied "
                    "data. If the data does not support a "
                    "claim, explicitly say that it cannot "
                    "be determined."
                ),
            ),
        )

        # ----------------------------------------------------
        # Check response
        # ----------------------------------------------------

        if not response.text:

            return (
                None,
                "Gemini returned an empty response."
            )

        return response.text, None

    except Exception as exc:

        return (
            None,
            f"Gemini request failed: {exc}"
        )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title(
    "🏆 AI Vendor Selection & Procurement Recommender"
)

st.caption(
    "Data-driven vendor ranking with AI-assisted "
    "explanation and trade-off analysis"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # Evaluation weights
    # --------------------------------------------------------

    st.header("1. Evaluation Weights")

    st.write(
        "Set the relative importance of each criterion. "
        "The total must equal 100%."
    )

    weights = {}

    for c in CRITERIA:

        weights[c] = st.slider(
            c,
            min_value=0,
            max_value=100,
            value=DEFAULT_WEIGHTS[c],
            step=5,
            key=f"w_{c}"
        )

    total = sum(weights.values())

    if total != 100:

        st.error(
            f"Weights currently total {total}%. "
            "Adjust them to 100%."
        )

    else:

        st.success(
            "✓ Weights total 100%"
        )

    st.divider()

    # --------------------------------------------------------
    # Gemini settings
    # --------------------------------------------------------

    st.header("2. AI Analysis")

    model_name = st.selectbox(
        "Gemini Model",
        [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ],
        index=0
    )

    st.caption(
        "Gemini is used only to explain the calculated "
        "vendor ranking. The ranking itself is generated "
        "by the application's scoring engine."
    )


# ============================================================
# VENDOR DATA INPUT
# ============================================================

st.subheader("3. Vendor Data")

col1, col2 = st.columns([1, 1])


# ------------------------------------------------------------
# Upload CSV
# ------------------------------------------------------------

with col1:

    uploaded = st.file_uploader(
        "Upload Vendor CSV",
        type=["csv"],
        help=(
            "CSV must contain Vendor plus the six "
            "evaluation criteria."
        )
    )


# ------------------------------------------------------------
# Load sample data
# ------------------------------------------------------------

with col2:

    if st.button(
        "Load Sample Dataset",
        use_container_width=True
    ):

        st.session_state.vendor_df = SAMPLE.copy()


# ------------------------------------------------------------
# Initialize dataset
# ------------------------------------------------------------

if "vendor_df" not in st.session_state:

    st.session_state.vendor_df = SAMPLE.copy()


# ------------------------------------------------------------
# Process uploaded file
# ------------------------------------------------------------

if uploaded is not None:

    try:

        st.session_state.vendor_df = pd.read_csv(
            uploaded
        )

    except Exception as exc:

        st.error(
            f"Could not read the CSV: {exc}"
        )


# ============================================================
# EDITABLE VENDOR TABLE
# ============================================================

df = st.data_editor(
    st.session_state.vendor_df,
    num_rows="dynamic",
    use_container_width=True,
    key="vendor_editor"
)

st.session_state.vendor_df = df


# ============================================================
# SCORING GUIDE
# ============================================================

with st.expander("📘 Scoring Guide"):

    st.markdown(
        """
        - **1 = Very weak performance**
        - **10 = Excellent performance**
        - For **Cost**, a higher score means a more
          competitive/lower cost.
        - For all other criteria, higher is better.
        - The final score is the weighted sum of the
          six criterion scores.
        """
    )


# ============================================================
# VALIDATE DATA
# ============================================================

errors = validate(df)

if errors:

    st.error(
        "Please fix the following data issues "
        "before analysis:"
    )

    for e in errors:

        st.write(
            f"• {e}"
        )

    st.stop()


# ============================================================
# VALIDATE WEIGHTS
# ============================================================

if total != 100:

    st.warning(
        "Set weights to exactly 100% to run the analysis."
    )

    st.stop()


# ============================================================
# CALCULATE VENDOR RANKING
# ============================================================

ranked = score_vendors(
    df,
    weights
)


# ============================================================
# VENDOR RANKING SECTION
# ============================================================

st.subheader("4. Vendor Ranking")

leader = ranked.iloc[0]


# ------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------

metric_cols = st.columns(3)

metric_cols[0].metric(
    "🏆 Recommended Vendor",
    leader["Vendor"]
)

metric_cols[1].metric(
    "Overall Score",
    f"{leader['Overall Score']:.2f}/10"
)

metric_cols[2].metric(
    "Vendors Evaluated",
    len(ranked)
)


# ------------------------------------------------------------
# Ranking table
# ------------------------------------------------------------

display_columns = [
    "Rank",
    "Vendor",
    "Overall Score"
] + CRITERIA

st.dataframe(
    ranked[display_columns]
    .style
    .format(
        {"Overall Score": "{:.2f}"}
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# OVERALL SCORE CHART
# ============================================================

chart = px.bar(
    ranked,
    x="Vendor",
    y="Overall Score",
    title="Overall Vendor Score",
    text="Overall Score"
)

chart.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside"
)

chart.update_yaxes(
    range=[0, 10.8]
)

st.plotly_chart(
    chart,
    use_container_width=True
)


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

st.subheader("5. What-if Analysis")

st.write(
    "Change the weights in the sidebar and observe how "
    "vendor rankings change when procurement priorities "
    "change."
)

st.info(
    "Example: Increasing the Cost weight may favor a "
    "lower-cost vendor, while increasing Quality or "
    "Reliability may favor a different supplier."
)


# ============================================================
# AI PROCUREMENT INSIGHT
# ============================================================

st.subheader("6. AI Procurement Insight")

st.write(
    "Gemini interprets the calculated ranking and explains "
    "the recommendation, trade-offs and risks."
)


if st.button(
    "🤖 Generate AI Recommendation",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Generating grounded procurement analysis..."
    ):

        explanation, error = get_ai_explanation(
            ranked,
            weights,
            model_name
        )

    if explanation:

        st.success(
            "AI analysis generated successfully."
        )

        st.markdown(
            explanation
        )

    else:

        st.warning(
            error
        )

        st.info(
            "The deterministic vendor ranking above "
            "remains valid even when the AI explanation "
            "is unavailable."
        )


# ============================================================
# DECISION SUPPORT DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Decision-support prototype: final supplier selection "
    "should remain subject to human review, commercial "
    "negotiation, compliance checks and organizational "
    "procurement policy."
)
