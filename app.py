import os
import sys
import joblib
import pandas as pd
import streamlit as st

# Ensure 'src/' is on the Python path so features.py & IQROutlierCapper load cleanly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from features import engineer_features

st.set_page_config(
    page_title="Project Failure Prediction System",
    page_icon="📊",
    layout="wide"
)


@st.cache_resource
def load_artifacts(model_filename: str):
    preprocessor = joblib.load(os.path.join("models", "preprocessor.joblib"))
    model = joblib.load(os.path.join("models", model_filename))
    return preprocessor, model


st.title("📊 Project Failure Prediction System")
st.markdown(
    "Enter project parameters in the sidebar to compute **Earned Value & Risk indicators** "
    "and predict the probability of project failure in real time."
)

# Sidebar: Model Selection & Project Inputs
st.sidebar.header("⚙️ Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    options=[
        ("Random Forest (Best ROC-AUC: 0.819)", "random_forest.joblib"),
        ("Logistic Regression (Highest Recall: 75%)", "logistic_regression.joblib"),
        ("XGBoost (ROC-AUC: 0.794)", "xgboost.joblib"),
    ],
    format_func=lambda x: x[0]
)

st.sidebar.header("📋 Project Parameters")
domain = st.sidebar.selectbox(
    "Project Domain",
    ["Software Engineering", "Agile IT", "Enterprise Infrastructure", "Startup Product"]
)
methodology = st.sidebar.selectbox(
    "Methodology",
    ["Agile", "Scrum", "Kanban", "Hybrid", "Waterfall"]
)

project_budget = st.sidebar.number_input("Planned Budget ($)", min_value=10000, max_value=5000000, value=250000, step=10000)
actual_cost = st.sidebar.number_input("Current / Projected Cost ($)", min_value=10000, max_value=10000000, value=290000, step=10000)

estimated_timeline = st.sidebar.slider("Estimated Timeline (Months)", 2.0, 36.0, 10.0, 0.5)
actual_duration = st.sidebar.slider("Current / Projected Duration (Months)", 2.0, 48.0, 12.0, 0.5)

complexity_score = st.sidebar.slider("Complexity Score (1–10)", 1.0, 10.0, 6.5, 0.1)
change_requests = st.sidebar.slider("Change Request Frequency", 0, 25, 7)
team_size = st.sidebar.slider("Team Size", 3, 50, 12)
stakeholder_count = st.sidebar.slider("Stakeholder Count", 2, 30, 8)
team_experience = st.sidebar.slider("Avg. Team Experience (Years)", 0.5, 15.0, 4.0, 0.5)
defect_density = st.sidebar.slider("Defect Density", 0.0, 15.0, 2.0, 0.1)
sentiment_score = st.sidebar.slider("Team Communication Sentiment (-1 to +1)", -1.0, 1.0, 0.1, 0.05)

# Assemble single-row DataFrame matching training schema
input_df = pd.DataFrame([{
    "domain": domain,
    "methodology": methodology,
    "project_budget": float(project_budget),
    "actual_cost": float(actual_cost),
    "estimated_timeline_months": float(estimated_timeline),
    "actual_duration_months": float(actual_duration),
    "complexity_score": float(complexity_score),
    "team_size": int(team_size),
    "team_experience_years": float(team_experience),
    "stakeholder_count": int(stakeholder_count),
    "change_request_frequency": int(change_requests),
    "defect_density": float(defect_density),
    "team_sentiment_score": float(sentiment_score)
}])

# Engineer PM Features & Predict
featured_df = engineer_features(input_df)
preprocessor, model = load_artifacts(model_choice[1])
X_processed = preprocessor.transform(featured_df)

failure_prob = float(model.predict_proba(X_processed)[0][1])
prediction = int(model.predict(X_processed)[0])

# Display Live Engineered PM Indicators
st.subheader("1. Engineered Project Management Indicators")
c1, c2, c3, c4 = st.columns(4)

cpi = featured_df["cpi"].iloc[0]
spi = featured_df["spi"].iloc[0]
comm_complex = int(featured_df["communication_complexity"].iloc[0])
mismatch = featured_df["methodology_mismatch_score"].iloc[0]

c1.metric("Cost Performance Index (CPI)", f"{cpi:.2f}", delta="Over Budget" if cpi < 1.0 else "On/Under Budget", delta_color="inverse" if cpi < 1.0 else "normal")
c2.metric("Schedule Performance Index (SPI)", f"{spi:.2f}", delta="Behind Schedule" if spi < 1.0 else "On/Ahead Schedule", delta_color="inverse" if spi < 1.0 else "normal")
c3.metric("Communication Channels", f"{comm_complex}", help="PMBOK formula: n(n-1)/2")
c4.metric("Methodology Mismatch", f"{mismatch:.2f}", help="Higher values signal methodology-volatility friction")

st.divider()

# Display Prediction Verdict & Proactive Recommendations
st.subheader("2. Failure Risk Assessment")
left_col, right_col = st.columns([1, 2])

with left_col:
    st.metric("Predicted Failure Probability", f"{failure_prob * 100:.1f}%")
    st.progress(min(max(failure_prob, 0.0), 1.0))

    if failure_prob >= 0.50:
        st.error("🚨 HIGH RISK: Project is likely to fail without intervention.")
    elif failure_prob >= 0.30:
        st.warning("⚠️ MODERATE RISK: Monitor closely and review risk register.")
    else:
        st.success("✅ LOW RISK: Project indicators are currently healthy.")

with right_col:
    st.markdown("**Proactive Managerial Recommendations:**")
    recs = []
    if cpi < 0.95:
        recs.append("- **Cost Control:** CPI is below 0.95. Audit resource burn rate and freeze non-essential scope.")
    if spi < 0.95:
        recs.append("- **Schedule Recovery:** SPI is below 0.95. Consider fast-tracking critical path tasks or reallocating senior engineers.")
    if mismatch > 3.0:
        recs.append(f"- **Methodology Shift:** `{methodology}` shows high friction with frequent change requests ({change_requests}). Consider transitioning to Agile/Scrum sprints.")
    if sentiment_score < 0.0:
        recs.append("- **Team Alignment:** Negative sentiment score detected. Hold a retrospective to clear blockers and reduce stakeholder friction.")
    if not recs:
        recs.append("- Core Earned Value and risk indicators are within healthy thresholds. Maintain standard weekly monitoring.")

    for r in recs:
        st.markdown(r)