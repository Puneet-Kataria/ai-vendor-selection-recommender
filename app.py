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
