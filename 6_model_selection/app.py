import numpy as np
import streamlit as st
import pickle
import pandas as pd
import os

st.title("🏡 Gurgaon Property Price Predictor")

# Load DataFrame
file_path = "6_model_selection/df.pkl"
if os.path.exists(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
else:
    st.error(f"File not found: {file_path}")
    st.stop()

# Load Model
file_path2 = "6_model_selection/pipeline.pkl"
if os.path.exists(file_path2):
    with open(file_path2, "rb") as file:
        pipeline = pickle.load(file)
else:
    st.error(f"File not found: {file_path2}")
    st.stop()

df = pd.DataFrame(data)

st.header("📝 Select Your Inputs")

property_type = st.selectbox("Property type", ["house", "flate"])
sector = st.selectbox("Select sector", df['sector'].unique().tolist())
bedroom = st.selectbox("Number of Bedrooms", sorted(df['bedRoom'].unique().tolist()))
bathroom = st.selectbox("Number of Bathrooms", sorted(df['bathroom'].unique().tolist()))
balcony = st.selectbox("Number of Balconies", sorted(df['balcony'].unique().tolist()))
property_age = st.selectbox("Property Age", sorted(df['agePossession'].unique().tolist()))
builtup_area = float(st.number_input('Built-up Area (in sqft)', min_value=100.0))
servant_room = float(st.selectbox("Servant Room", [0, 1]))
store_room = float(st.selectbox("Store Room", [0, 1], key="store_room_key"))
furnishing_type = st.selectbox("Furnishing Type", sorted(df['furnishing_type'].unique().tolist()))
luxury_category = st.selectbox("Luxury Category", sorted(df['luxury_category'].unique().tolist()))
floor_category = st.selectbox("Floor Category", sorted(df['floor_category'].unique().tolist()))

if st.button("Predict"):
    input_data = [[
        property_type, sector, bedroom, bathroom, balcony,
        property_age, builtup_area, servant_room, store_room,
        furnishing_type, luxury_category, floor_category
    ]]

    columns = [
        'property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
        'agePossession', 'built_up_area', 'servant room', 'store room',
        'furnishing_type', 'luxury_category', 'floor_category'
    ]

    input_df = pd.DataFrame(input_data, columns=columns)

    # Make prediction
    try:
        price = np.expm1(pipeline.predict(input_df))
        upper_bound = price*100+15.0
        lower_bound = price*100-15.0
        st.success(f"🏷️ Predicted Price is Between: ₹{Lower_bound} L - ₹{upper_bound} L")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
