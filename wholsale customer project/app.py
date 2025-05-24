import streamlit as st
import pickle
import numpy as np

# Function to load model and scaler
def load_model_and_scaler(model_path, scaler_path):
    with open(r"D:\Machine learning\wholsale customer project\wholesale_model.pkl", 'rb') as model_file:
        model = pickle.load(model_file)
    with open(r"D:\Machine learning\wholsale customer project\scaler (1).pkl", 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Function to predict cluster
def predict_cluster(model, scaler, features):
    input_array = np.array([features]).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]
    return prediction

# Streamlit UI
st.title("Wholesale Customer Cluster Prediction")

st.markdown("Enter the following customer features:")

channel = st.selectbox("Channel", [1, 2])  # 1=Horeca, 2=Retail
region = st.selectbox("Region", [1, 2, 3])  # e.g., 1=Lisbon, 2=Oporto, 3=Other

fresh = st.number_input("Fresh", min_value=0, value=12669)
milk = st.number_input("Milk", min_value=0, value=9656)
grocery = st.number_input("Grocery", min_value=0, value=7561)
frozen = st.number_input("Frozen", min_value=0, value=214)
detergents = st.number_input("Detergents_Paper", min_value=0, value=2674)
delicassen = st.number_input("Delicassen", min_value=0, value=1338)

if st.button("Predict Cluster"):
    try:
        model, scaler = load_model_and_scaler("wholesale_model.pkl", "scaler.pkl")
        input_features = [channel, region, fresh, milk, grocery, frozen, detergents, delicassen]
        result = predict_cluster(model, scaler, input_features)
        st.success(f"Predicted Cluster: {result}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
