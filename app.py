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
        # Start with all-zero feature dict
        input_data = {col: 0 for col in feature_columns}

        # Numeric features
        input_data["Age"] = int(request.form["Age"])
        input_data["DailyRate"] = int(request.form["DailyRate"])
        input_data["DistanceFromHome"] = int(request.form["DistanceFromHome"])
        input_data["Education"] = int(request.form["Education"])
        input_data["JobLevel"] = int(request.form["JobLevel"])
        input_data["MonthlyIncome"] = int(request.form["MonthlyIncome"])
        input_data["TotalWorkingYears"] = int(request.form["TotalWorkingYears"])
        input_data["YearsAtCompany"] = int(request.form["YearsAtCompany"])

        # Gender (one-hot)
        if request.form["Gender"] == "Male":
            input_data["Gender_Male"] = 1

        # OverTime (one-hot)
        if request.form["OverTime"] == "Yes":
            input_data["OverTime_Yes"] = 1

        # Department (one-hot)
        dept = request.form["Department"]
        if dept == "Sales":
            input_data["Department_Sales"] = 1
        elif dept == "Research & Development":
            input_data["Department_Research & Development"] = 1
        elif dept == "Human Resources":
            input_data["Department_Human Resources"] = 1

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)[0]

        result = (
            "✅ Employee is likely to STAY in the company"
            if prediction == 0
            else "❌ Employee is likely to LEAVE the company"
        )

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template(
            "index.html",
            prediction="⚠️ Error: Please check your input values"
        )


if __name__ == "__main__":
    app.run()

