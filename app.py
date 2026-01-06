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
    return joblib.load("random_forest_model.pkl")

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
# UI
# -----------------------------------
st.title("Employee Attrition Prediction")
st.write("Fill the employee details and click **Predict**")

# -----------------------------------
# User Inputs
# -----------------------------------
age = st.number_input("Age", min_value=18, max_value=65, value=30)
daily_rate = st.number_input("Daily Rate", value=800)
distance = st.number_input("Distance From Home", value=10)
education = st.selectbox("Education Level (1–5)", [1, 2, 3, 4, 5])
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
monthly_income = st.number_input("Monthly Income", value=5000)
total_years = st.number_input("Total Working Years", value=5)
years_company = st.number_input("Years at Company", value=3)

gender = st.selectbox("Gender", ["Male", "Female"])
overtime = st.selectbox("Over Time", ["Yes", "No"])
department = st.selectbox(
    "Department",
    ["Sales", "Research & Development", "Human Resources"]
)

# -----------------------------------
# Prepare input dictionary
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

# Gender
if gender == "Male":
    for col in feature_columns:
        if col.lower() == "gender_male":
            input_data[col] = 1

# OverTime
if overtime == "Yes":
    for col in feature_columns:
        if col.lower() == "overtime_yes":
            input_data[col] = 1

# Department (AUTO-MATCH SAFE)
dept_key = department.replace(" & ", "_").replace(" ", "_").lower()
for col in feature_columns:
    if col.lower().startswith("department_") and dept_key in col.lower():
        input_data[col] = 1

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
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error("❌ Employee is likely to **LEAVE** the company")
        else:
            st.success("✅ Employee is likely to **STAY** in the company")

        st.info(f"📊 Attrition Probability: **{probability * 100:.2f}%**")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
