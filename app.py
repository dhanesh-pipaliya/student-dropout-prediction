import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and preprocessing tools
model, scaler, le_dict, feature_order = joblib.load("rf_dropout_model.pkl")

st.title("🎓 Student Dropout Risk Predictor")
st.write("Predict whether a student is at risk of dropping out using key features.")

# Define user input fields
inputs = {}
inputs['age'] = st.slider("Age", 15, 22)
inputs['absences'] = st.slider("Absences", 0, 100)
inputs['studytime'] = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
inputs['failures'] = st.slider("Past Class Failures", 0, 4)
inputs['internet'] = st.selectbox("Internet at Home", ['yes', 'no'])
inputs['schoolsup'] = st.selectbox("School Support", ['yes', 'no'])
inputs['goout'] = st.slider("Going Out (Social Activity)", 1, 5)
inputs['health'] = st.slider("Health (1 = poor, 5 = excellent)", 1, 5)
inputs['G1'] = st.slider("Grade in Period 1 (G1)", 0, 20)
inputs['G2'] = st.slider("Grade in Period 2 (G2)", 0, 20)

# Helper function to get a default known value from encoder
def get_default_encoded_value(le):
    return le.transform([le.classes_[0]])[0]

# Prepare the full feature input
input_data = {}
for col in feature_order:
    if col in inputs:
        val = inputs[col]
        if col in le_dict:
            try:
                val = le_dict[col].transform([val])[0]
            except ValueError:
                val = get_default_encoded_value(le_dict[col])
        input_data[col] = val
    elif col in le_dict:
        input_data[col] = get_default_encoded_value(le_dict[col])
    else:
        input_data[col] = 0  # default numeric value

# Convert to DataFrame
input_df = pd.DataFrame([input_data])[feature_order]

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Dropout Risk"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result:")
    if pred == 1:
        st.error(f"⚠️ High Risk of Dropping Out (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk (Probability: {prob:.2f})")
