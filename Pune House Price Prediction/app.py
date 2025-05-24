import streamlit as st
import pickle
import numpy as np

# Load model and column data
with open("D:\\Machine learning\\Pune House Price Prediction\\model.pkl", "rb") as f:
    model = pickle.load(f)

with open("D:\\Machine learning\\Pune House Price Prediction\\columns.pkl", "rb") as f:
    data_columns = pickle.load(f)

# Extract categories from columns.pkl
locations = data_columns['locations']
area_types = data_columns['area_types']
availabilities = data_columns['availabilities']
all_columns = data_columns['data_columns']

# Debug (optional): Uncomment to inspect column names
# st.write("Available Columns:", all_columns)

# Find correct sqft column name
sqft_column = 'sqft'
if sqft_column not in all_columns:
    # Try fallback names
    for candidate in ['total_sqft', 'new_total_sqft', 'sqft']:
        if candidate in all_columns:
            sqft_column = candidate
            break
    else:
        st.error("❌ 'sqft' column not found in data_columns!")
        st.stop()

# Prediction function
def predict_price(location, area_type, availability, new_total_sqft, bath, bhk):
    x = np.zeros(len(all_columns))
    
    # Fill numerical inputs
    x[all_columns.index(sqft_column)] = new_total_sqft
    x[all_columns.index('bath')] = bath
    x[all_columns.index('bhk')] = bhk

    # One-hot encoded categorical values
    loc_col = f"location_{location}"
    area_col = f"area_type_{area_type}"
    avail_col = f"availability_{availability}"

    if loc_col in all_columns:
        x[all_columns.index(loc_col)] = 1
    if area_col in all_columns:
        x[all_columns.index(area_col)] = 1
    if avail_col in all_columns:
        x[all_columns.index(avail_col)] = 1

    price = model.predict([x])[0]
    return round(price, 2)

# Streamlit UI
st.set_page_config(page_title="Pune House Price Prediction", layout="centered")
st.title("🏠 Pune House Price Prediction")
st.markdown("Enter property details to get the estimated price.")

location = st.selectbox("Select Location", locations)
area_type = st.selectbox("Select Area Type", area_types)
availability = st.selectbox("Select Availability", availabilities)
new_total_sqft = st.number_input("Total Square Feet Area", min_value=300, max_value=10000, value=1000)
bath = st.slider("Number of Bathrooms", 1, 10, 2)
bhk = st.slider("Number of BHKs", 1, 10, 2)

if st.button("Predict Price"):
    result = predict_price(location, area_type, availability, new_total_sqft, bath, bhk)
    st.success(f"🏡 Estimated Price: ₹{result} Lakhs")
