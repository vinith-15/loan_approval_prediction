from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("model/loan_model.pkl", "rb"))
encoder = pickle.load(open("model/encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = int(request.form["age"])
        income = float(request.form["income"])
        loan = float(request.form["loan"])
        credit = int(request.form["credit"])
        employment = request.form["employment"]

        # Encode employment
        emp = encoder.transform([employment])[0]

        data = np.array([[age, income, loan, credit, emp]])

        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        if prediction == 1:
            result = f"Loan Approved ✅ (Confidence: {prob:.2f})"
        else:
            result = f"Loan Rejected ❌ (Confidence: {prob:.2f})"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))