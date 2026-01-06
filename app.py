import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load model and feature columns
# -----------------------------
model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")

st.title("Employee Attrition Prediction")
st.write("Fill the employee details and click **Predict**")

# -----------------------------
# USER INPUTS (IMPORTANT ONES)
# -----------------------------
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

# -----------------------------
# CREATE INPUT DICTIONARY
# -----------------------------
input_data = {col: 0 for col in feature_columns}

# Numeric features
input_data["Age"] = age
input_data["DailyRate"] = daily_rate
input_data["DistanceFromHome"] = distance
input_data["Education"] = education
input_data["JobLevel"] = job_level
input_data["MonthlyIncome"] = monthly_income
input_data["TotalWorkingYears"] = total_years
input_data["YearsAtCompany"] = years_company

# One-hot encoding (manual)
if gender == "Male":
    input_data["Gender_Male"] = 1

if overtime == "Yes":
    input_data["OverTime_Yes"] = 1

if department == "Sales":
    input_data["Department_Sales"] = 1
elif department == "Research & Development":
    input_data["Department_Research & Development"] = 1
elif department == "Human Resources":
    input_data["Department_Human Resources"] = 1

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("❌ Employee is likely to LEAVE the company")
    else:
        st.success("✅ Employee is likely to STAY in the company")
