import streamlit as st
import joblib
import pandas as pd

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="centered"
)

# -----------------------------------
# Load model & features safely
# -----------------------------------
@st.cache_resource
def load_model():
    return joblib.load("random_forest_modell.pkl")  # ✅ fixed name

@st.cache_resource
def load_features():
    return joblib.load("feature_columns.pkl")

try:
    model = load_model()
    feature_columns = load_features()
except Exception as e:
    st.error("❌ Failed to load model or feature columns")
    st.exception(e)
    st.stop()

# -----------------------------------
# Helper for one-hot encoding (SAFE)
# -----------------------------------
def set_one_hot(input_data, prefix, value):
    key = f"{prefix}_{value}".replace(" & ", "_").replace(" ", "_").lower()
    for col in feature_columns:
        if col.lower() == key:
            input_data[col] = 1

# -----------------------------------
# UI
# -----------------------------------
st.title("Employee Attrition Prediction")
st.write("Fill employee details and click **Predict**")

# -----------------------------------
# User Inputs (GROUPED & SIMPLE)
# -----------------------------------

with st.expander("👤 Personal Information", expanded=True):
    age = st.number_input("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    distance = st.number_input("Distance From Home (km)", 0, 50, 10)

with st.expander("💼 Job Information"):
    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )
    job_level = st.selectbox("Job Level (1–5)", [1, 2, 3, 4, 5])
    education = st.selectbox("Education Level (1–5)", [1, 2, 3, 4, 5])
    overtime = st.selectbox("Over Time", ["Yes", "No"])

with st.expander("💰 Compensation & Experience"):
    daily_rate = st.number_input("Daily Rate", 100, 2000, 800)
    monthly_income = st.number_input("Monthly Income", 1000, 50000, 5000)
    total_years = st.number_input("Total Working Years", 0, 40, 5)
    years_company = st.number_input("Years at Company", 0, 40, 3)

# -----------------------------------
# Prepare input dictionary (IMPORTANT)
# -----------------------------------
input_data = {col: 0 for col in feature_columns}

# Numeric features
input_data.update({
    "Age": age,
    "DailyRate": daily_rate,
    "DistanceFromHome": distance,
    "Education": education,
    "JobLevel": job_level,
    "MonthlyIncome": monthly_income,
    "TotalWorkingYears": total_years,
    "YearsAtCompany": years_company
})

# Binary features
if overtime == "Yes":
    set_one_hot(input_data, "OverTime", "Yes")

# One-hot categorical
set_one_hot(input_data, "Gender", gender)
set_one_hot(input_data, "Department", department)

# -----------------------------------
# Create DataFrame (CLOUD SAFE)
# -----------------------------------
input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# -----------------------------------
# Prediction
# -----------------------------------
if st.button("Predict"):
    try:
        proba = model.predict_proba(input_df)[0][1]

        # 🔥 threshold tuning for imbalance
        prediction = 1 if proba >= 0.4 else 0

        if prediction == 1:
            st.error("❌ Employee is likely to **LEAVE** the company")
        else:
            st.success("✅ Employee is likely to **STAY** in the company")

        st.info(f"📊 Attrition Probability: **{proba * 100:.2f}%**")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
