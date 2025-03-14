import streamlit as st
import pickle
import numpy as np

import joblib
import sklearn

import pickle

def load_model(model_path, scaler_path):
    with open(r'D:\Machine learning\heart disease dataset\heart_disease_model.pkl', "rb") as model_file:
        model = pickle.load(model_file, fix_imports=True, encoding="latin1", errors="ignore")

    with open(r'D:\Machine learning\heart disease dataset\heart_disease_scaler.pkl', "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler

def infer_heart_disease(model, scaler, input_features):
    """Perform inference on user input."""
    input_array = np.array([input_features]).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    return "Disease Detected" if prediction[0] == 1 else "No Disease"

# Streamlit UI
st.title("Heart Disease Prediction App")

st.sidebar.header("Enter Patient Details")
age = st.sidebar.number_input("Age", 1, 120, 45)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol", 100, 600, 233)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.sidebar.slider("Resting ECG (0-2)", 0, 2, 1)
thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", 0.0, 6.2, 2.3)
slope = st.sidebar.slider("Slope of Peak Exercise ST Segment", 0, 2, 1)
ca = st.sidebar.slider("Number of Major Vessels (0-4)", 0, 4, 0)
thal = st.sidebar.slider("Thalassemia (0-3)", 0, 3, 1)

if st.sidebar.button("Predict"):
    input_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    model, scaler = load_model("heart_disease_model.pkl", "heart_disease_scaler.pkl")
    result = infer_heart_disease(model, scaler, input_features)
    st.success(f"Prediction: {result}")
