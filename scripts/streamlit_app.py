# =====================================
# streamlit_app.py - Heart Disease Prediction & Exploration
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ------------------------------
# Load your trained model
# ------------------------------
model_path = os.path.join(os.path.dirname(__file__), '../models/heart_disease_model.pkl')

if not os.path.exists(model_path):
    st.error("Model file not found! Please check that 'heart_disease_model.pkl' exists in the models folder.")
    st.stop()

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# ------------------------------
# Streamlit App UI
# ------------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Please input the following parameters:")

# ------------------------------
# Input features
# ------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x==1 else "Female")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting ECG results", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of ST segment", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels colored by fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

# ------------------------------
# Collect input into a DataFrame
# ------------------------------
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]],
                          columns=["age", "sex", "cp", "trestbps", "chol", "fbs",
                                   "restecg", "thalach", "exang", "oldpeak", "slope",
                                   "ca", "thal"])

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    result_text = "❤️ Heart Disease Detected" if prediction == 1 else "✅ No Heart Disease"
    st.subheader(f"Prediction: {result_text}")
    st.write(f"Prediction Probability: No Disease: {prediction_proba[0]*100:.2f}%, Disease: {prediction_proba[1]*100:.2f}%")

    # ------------------------------
    # Plot prediction result
    # ------------------------------
    fig, ax = plt.subplots()
    sns.barplot(x=["No Disease", "Disease"], y=prediction_proba, palette="coolwarm", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

# ------------------------------
# Display input data
# ------------------------------
st.subheader("Patient Input Data")
st.write(input_data)
