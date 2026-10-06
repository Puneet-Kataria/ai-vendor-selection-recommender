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
            "Vendor names cannot be blank."
        )

    if (
        df["Vendor"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        errors.append(
            "Vendor names cannot be blank."
        )

    # Check duplicate vendor names
    vendor_names = (
        df["Vendor"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if vendor_names.duplicated().any():
        errors.append(
            "Duplicate vendor names found. "
            "Each vendor must be unique."
        )

    # Check criterion values
    for c in CRITERIA:

        numeric = pd.to_numeric(
            df[c],
            errors="coerce"
        )

        if numeric.isna().any():

            errors.append(
                f"{c}: all values must be numeric "
                "and between 1 and 10."
            )

        elif (
            (numeric < 1) |
            (numeric > 10)
        ).any():

            errors.append(
                f"{c}: scores must be between 1 and 10."
            )

    return errors


# ============================================================
# VENDOR SCORING
# ============================================================

def score_vendors(df, weights):

    out = df.copy()

    # Convert criteria to numeric
    for c in CRITERIA:

        out[c] = pd.to_numeric(
            out[c],
            errors="coerce"
        )

    # Weighted score
    out["Overall Score"] = sum(
        out[c] * (weights[c] / 100)
        for c in CRITERIA
    )

    # Rank vendors
    out = (
        out
        .sort_values(
            "Overall Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    out.insert(
        0,
        "Rank",
        range(1, len(out) + 1)
    )

    return out


# ============================================================
# GEMINI AI ANALYSIS
# ============================================================

def get_ai_explanation(ranked, weights):

    # --------------------------------------------------------
    # Read Gemini API key securely from Streamlit Secrets
    # --------------------------------------------------------

    try:

        api_key = st.secrets["GEMINI_API_KEY"]

    except Exception:

        api_key = ""

    # --------------------------------------------------------
    # Check API key
    # --------------------------------------------------------

    if not api_key:

        return (
            None,
            "Gemini API key is not configured. "
            "Please add GEMINI_API_KEY under "
            "Streamlit → Settings → Secrets."
        )

    # --------------------------------------------------------
    # Check package
    # --------------------------------------------------------

    if genai is None or types is None:

        return (
            None,
            "The google-genai package is not installed. "
            "Please add google-genai to requirements.txt "
            "and redeploy the application."
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
    # Prepare vendor records
    # --------------------------------------------------------

    records = (
        ranked[
            [
                "Rank",
                "Vendor",
                "Overall Score"
            ] + CRITERIA
        ]
        .round(2)
        .to_dict(orient="records")
    )

    # --------------------------------------------------------
    # Grounded AI prompt
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
   locations, reputation, capacity, delivery history,
   market reputation or any other external information.

4. Do NOT change the calculated ranking.

5. Do NOT independently create a different ranking.

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
    runner["Vendor"]
    if runner is not None
    else "No runner-up available"
}

RUNNER-UP SCORE:

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
    # Gemini API request with bounded retry handling
    # --------------------------------------------------------
    #
    # Gemini can temporarily return 503 UNAVAILABLE when the
    # model is experiencing high demand. We retry only transient
    # server-side errors, using exponential backoff with jitter.
    # --------------------------------------------------------

    import random
    import time

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            client = genai.Client(
                api_key=api_key
            )

            response = client.models.generate_content(

                model=GEMINI_MODEL,

                contents=prompt,

                config=types.GenerateContentConfig(

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

            # ------------------------------------------------
            # Check response
            # ------------------------------------------------

            if not response.text:

                return (
                    None,
                    "Gemini returned an empty response."
                )

            return response.text, None

        except Exception as exc:

            error_text = str(exc)

            # Retry only temporary server-side errors.
            transient_error = any(
                code in error_text
                for code in ["500", "502", "503", "504"]
            )

            if transient_error and attempt < max_attempts - 1:

                # Exponential backoff with a small random jitter.
                # Attempt 1: ~2–3 seconds
                # Attempt 2: ~4–5 seconds
                wait_time = (
                    (2 ** (attempt + 1))
                    + random.uniform(0, 1)
                )

                time.sleep(wait_time)
                continue

            # ------------------------------------------------
            # Final failure message
            # ------------------------------------------------

            if "503" in error_text or "UNAVAILABLE" in error_text:

                return (
                    None,
                    "Gemini is temporarily unavailable because "
                    "the model is experiencing high demand. "
                    "The application automatically retried the "
                    "request, but Gemini did not become available. "
                    "Please try the AI recommendation again in a "
                    "few moments."
                )

            return (
                None,
                f"Gemini request failed: {error_text}"
            )

    return (
        None,
        "Gemini could not generate the recommendation after "
        "multiple attempts."
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
    # AI Analysis
    # --------------------------------------------------------

    st.header("2. AI Analysis")

    st.success(
        "✓ Gemini 3.8 Flash"
    )

    st.caption(
        "Gemini is used to explain the calculated vendor "
        "ranking, trade-offs and risks. The ranking itself "
        "is generated by the application's scoring engine."
    )


# ============================================================
# VENDOR DATA
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
# Load sample dataset
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
# Process uploaded CSV
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
# EDITABLE DATA TABLE
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
        **Score range:** 1–10

        - **1 = Very weak performance**
        - **10 = Excellent performance**
        - For **Cost**, a higher score means a more
          competitive/lower cost.
        - For all other criteria, higher is better.
        - The final score is calculated using the
          weighted sum of all six criteria.
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
# CALCULATE RANKING
# ============================================================

ranked = score_vendors(
    df,
    weights
)


# ============================================================
# VENDOR RANKING
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
            weights
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
