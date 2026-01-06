import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

@st.cache_resource
def load_features():
    return joblib.load("feature_columns.pkl")

model = load_model()
feature_columns = load_features()

st.title("Employee Attrition Prediction")
st.write("Fill the employee details and click **Predict**")

# Inputs
age = st.number_input("Age", 18, 65, 30)
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

# Create input
input_data = {col: 0 for col in feature_columns}

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

input_df = pd.DataFrame([input_data])
input_df = input_df[feature_columns]   # 🔒 SAFETY LINE

if st.button("Predict"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("❌ Employee is likely to LEAVE the company")
    else:
        st.success("✅ Employee is likely to STAY in the company")
