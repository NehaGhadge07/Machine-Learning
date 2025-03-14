import pickle
import numpy as np
import streamlit as st

def load_model():
    with open(r'D:\Machine learning\Diabetes\daibetes_detection.pkl', "rb") as model_file:
        model = pickle.load(model_file)
    with open(r'D:\Machine learning\Diabetes\daibetes_detection_scaler.pkl', "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

def predict_diabetes(model, scaler, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)
    return "Diabetes Detected" if prediction[0] == 1 else "Diabetes Not Detected"

def main():
    st.title("Diabetes Prediction App")
    st.write("Enter the required details to check for diabetes")
    
    # Input fields
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    
    if st.button("Predict"):
        model, scaler = load_model()
        input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
        result = predict_diabetes(model, scaler, input_data)
        st.success(result)

if __name__ == "__main__":
    main()
