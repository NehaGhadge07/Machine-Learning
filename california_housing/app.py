import streamlit as st
import numpy as np
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    with open(r'D:\Machine learning\california_housing\california_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

def predict_house_value(input_features):
    """
    Predicts median house value using the trained model.
    """
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# Streamlit UI
st.title("🏠 California House Price Prediction")
st.markdown("Enter the features below to predict the **median house value**.")

# Input fields
med_inc = st.number_input("Median Income (MedInc)", min_value=0.0, value=8.3252)
house_age = st.number_input("House Age", min_value=0.0, value=41.0)
avg_rooms = st.number_input("Average Rooms (AveRooms)", min_value=0.0, value=6.9841)
avg_bedrooms = st.number_input("Average Bedrooms (AveBedrms)", min_value=0.0, value=1.0238)
population = st.number_input("Population", min_value=0.0, value=322.0)
avg_occupancy = st.number_input("Average Occupancy (AveOccup)", min_value=0.0, value=2.5556)
latitude = st.number_input("Latitude", value=37.88)
longitude = st.number_input("Longitude", value=-122.23)

# Predict button
if st.button("Predict House Value"):
    features = [med_inc, house_age, avg_rooms, avg_bedrooms, population, avg_occupancy, latitude, longitude]
    prediction = predict_house_value(features)
    st.success(f"🏡 Predicted Median House Value: **${prediction:,.2f}**")
