import os
import pickle
import streamlit as st
import numpy as np

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction Tool")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, 'Model', 'house_model.pkl')

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"❌ Model file not found! Expected it at: {model_path}")
    st.stop()

st.header("Property Details")
col1, col2 = st.columns(2)
with col1:
    sqft = st.number_input("Total Square Footage", min_value=300, max_value=10000, value=1500)
    bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=8, value=3)
    bathrooms = st.slider("Number of Bathrooms", min_value=1.0, max_value=6.0, value=2.0, step=0.5)
with col2:
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2010)
    location_score = st.slider("Neighborhood Rating (1-10)", min_value=1.0, max_value=10.0, value=5.0)

if st.button("Calculate Estimated Value", type="primary"):
    input_features = np.array([[sqft, bedrooms, bathrooms, year_built, location_score]])
    predicted_price = model.predict(input_features)[0]
    st.success(f"💰 Estimated Market Value: **${predicted_price:,.2f}**")
