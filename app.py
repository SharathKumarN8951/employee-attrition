from flask import Flask, render_template, request
import joblib
import pandas as pd

# ✅ CREATE APP FIRST
app = Flask(__name__)

# Load model & feature columns
model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Initialize all features to 0
        input_data = {col: 0 for col in feature_columns}

        # Safe numeric inputs
        input_data["Age"] = int(request.form.get("age", 0))
        input_data["MonthlyIncome"] = int(request.form.get("monthly_income", 0))
        input_data["DistanceFromHome"] = int(request.form.get("distance", 0))
        input_data["TotalWorkingYears"] = int(request.form.get("total_years", 0))
        input_data["YearsAtCompany"] = int(request.form.get("years_company", 0))

        # Categorical inputs
        if request.form.get("gender") == "Male":
            input_data["Gender_Male"] = 1

        if request.form.get("overtime") == "Yes":
            input_data["OverTime_Yes"] = 1

        # Predict
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        result = (
            "Employee is likely to LEAVE ❌"
            if prediction == 1
            else "Employee is likely to STAY ✅"
        )

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template(
            "index.html",
            prediction="⚠️ Error processing input. Please check all fields."
        )

if __name__ == "__main__":
    app.run()
