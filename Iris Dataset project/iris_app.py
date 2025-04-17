import streamlit as st
import numpy as np
import pickle

# Title
st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower's features to predict its species.")

# Load model and scaler
@st.cache_resource
def load_model_scaler():
    with open(r"D:\Machine learning\Iris Dataset project\scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(r"D:\Machine learning\Iris Dataset project\model.pkl", "rb") as f:
        model = pickle.load(f)
    return scaler, model

scaler, model = load_model_scaler()

# User input
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict"):
    input_data = [sepal_length, sepal_width, petal_length, petal_width]
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    iris_classes = ['setosa', 'versicolor', 'virginica']
    predicted_class = iris_classes[prediction[0]]

    st.subheader("🌼 Predicted Iris Species:")
    st.success(predicted_class.capitalize())
