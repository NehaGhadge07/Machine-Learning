# 🩺 Diabetes Prediction Web App

This is a machine learning-based web application built using **Streamlit** that predicts whether a person is likely to have diabetes based on input medical parameters.

## 📌 Overview

The application uses a trained classification model (stored as `.pkl` files) and standard scaler to preprocess user inputs and provide real-time diabetes prediction.

## 🧠 Features

- User-friendly interface built with Streamlit.
- Takes input on key health parameters such as:
  - Pregnancies
  - Glucose
  - Blood Pressure
  - Skin Thickness
  - Insulin
  - BMI
  - Diabetes Pedigree Function
  - Age
- Displays prediction result: **Diabetes Detected** or **Diabetes Not Detected**

## 🛠️ Technologies Used

- Python
- Streamlit
- scikit-learn
- NumPy
- Pickle (for model serialization)

## 📂 File Structure

```bash
├── diabetes_app.py                  # Main Streamlit app
├── daibetes_detection.pkl           # Trained ML model (not included here)
├── daibetes_detection_scaler.pkl    # StandardScaler object used for input preprocessing
├── README.md                        # Project description

Ensure you have all required dependencies:

pip install streamlit scikit-learn numpy
Place the following files in the root folder:

daibetes_detection.pkl
daibetes_detection_scaler.pkl

Run the app:
streamlit run diabetes_app.py
