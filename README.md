# AI Vendor Selection & Procurement Recommender

A Streamlit prototype for the end-term AI application project.

## What it does
- Accepts vendor data by CSV or manual editing.
- Validates missing, duplicate, non-numeric and out-of-range values.
- Lets the procurement user assign weights to six criteria.
- Calculates deterministic weighted vendor scores and rankings.
- Provides a visual ranking dashboard.
- Uses Gemini for a grounded natural-language explanation of the ranking, trade-offs and risks.
- Keeps the analytical ranking available if the Gemini API is unavailable.

## Run locally

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
5. Optional: provide `GEMINI_API_KEY` as an environment variable or paste a key in the sidebar for the demo.

## Data format

CSV columns must be:
`Vendor, Cost, Quality, Delivery, Reliability, Sustainability, Payment Terms`

Scores are 1–10. Higher is better. For Cost, a higher score means lower/more competitive cost.

## Suggested demo flow

1. Load sample dataset.
2. Show validation and scoring guide.
3. Explain default weights.
4. Show ranked vendors and chart.
5. Change Cost weight upward and show ranking sensitivity.
6. Generate Gemini explanation.
7. Demonstrate an edge case such as entering 15 for Quality; the app rejects it.
8. Explain human oversight and AI limitations.
