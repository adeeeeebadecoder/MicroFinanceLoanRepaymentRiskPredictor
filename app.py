import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("loan_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Micro-finance Loan Repayment Risk Predictor")

st.title("🏦 Micro-finance Loan Repayment Risk Predictor")

st.write("Enter applicant details to predict repayment risk.")

# ===== USER INPUT =====
principal = st.number_input("Loan Amount", 500, 10000, 3000)
terms = st.slider("Loan Term (weeks)", 1, 52, 12)
age = st.slider("Age", 18, 70, 30)

gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox(
    "Education",
    ["High School or Below", "College", "Bachelor Degree", "Master or Above"]
)
guarantor = st.selectbox("Guarantor Available?", ["Yes", "No"])

# ===== CREATE INPUT DATAFRAME =====
input_data = pd.DataFrame({
    "principal": [principal],
    "terms": [terms],
    "age": [age],
    "gender": [gender],
    "highest_education": [education],
    "guarantor": [guarantor]
})

# One-hot encode
input_encoded = pd.get_dummies(input_data, drop_first=True)

# Align with training columns
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# ===== PREDICTION BUTTON =====
if st.button("Predict Risk"):

    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    # ===== RISK SCORING SYSTEM =====
    risk_score = round((1 - probability) * 100)

    # Risk category
    if risk_score <= 30:
        risk_level = "LOW RISK ✅"
        decision = "Loan Approved"
        color = "green"

    elif risk_score <= 60:
        risk_level = "MEDIUM RISK ⚠️"
        decision = "Manual Review Needed"
        color = "orange"

    else:
        risk_level = "HIGH RISK ❌"
        decision = "Loan Rejected"
        color = "red"

    # ===== DISPLAY RESULTS =====
    st.subheader("Prediction Result")

    st.write(f"**Repayment Probability:** {round(probability*100,2)}%")
    st.write(f"**Risk Score:** {risk_score}/100")

    st.markdown(f"### Risk Level: :{color}[{risk_level}]")
    st.markdown(f"### Decision: :{color}[{decision}]")

    # Extra insight
    if risk_score > 60:
        st.warning("Applicant shows high default risk.")
    elif risk_score > 30:
        st.info("Applicant requires manual verification.")
    else:
        st.success("Applicant is safe for loan approval.")