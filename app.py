from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("lrR.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form.get("age", 0))
    hypertension = int(request.form.get("hypertension", 0))
    heart_disease = int(request.form.get("heart_disease", 0))
    avg_glucose_level = float(request.form.get("avg_glucose_level", 0))
    bmi = float(request.form.get("bmi", 0))

    gender_male = int(request.form.get("gender_male", 0))
    gender_other = int(request.form.get("gender_other", 0))
    ever_married_yes = int(request.form.get("ever_married_yes", 0))

    work_private = int(request.form.get("work_private", 0))
    work_self_employed = int(request.form.get("work_self_employed", 0))
    work_children = int(request.form.get("work_children", 0))
    work_never = int(request.form.get("work_never", 0))

    residence_urban = int(request.form.get("residence_urban", 0))

    smoke_former = int(request.form.get("smoke_former", 0))
    smoke_never = int(request.form.get("smoke_never", 0))
    smoke_current = int(request.form.get("smoke_current", 0))

    data = np.array([[
        age, hypertension, heart_disease, avg_glucose_level, bmi,
        gender_male, gender_other, ever_married_yes,
        work_never, work_private, work_self_employed, work_children,
        residence_urban, smoke_former, smoke_never, smoke_current
    ]])

    result = model.predict(data)
    proba = model.predict_proba(data)[0][1]

    prediction_value = (
        "No Stroke Risk" if result[0] == 0 else "High Stroke Risk"
    )

    return render_template(
        "index.html",
        prediction_value=prediction_value,
        prediction_proba=f"{proba:.2f}"
    )


if __name__ == "__main__":
    app.run(debug=True)

