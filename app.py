import os
import pickle
import streamlit as st
import pandas as pd

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction Tool")

# --- BULLETPROOF PATH RESOLUTION ---
# Get the absolute directory of the script running this code
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Option A: Look one directory up, then in 'Model' (Matches: project/Model/house_model.pkl)
path_option_a = os.path.join(os.path.dirname(SCRIPT_DIR), 'Model', 'house_model.pkl')

# Option B: Look in a 'Model' folder in the same directory (Matches: project/src/Model/house_model.pkl)
path_option_b = os.path.join(SCRIPT_DIR, 'Model', 'house_model.pkl')

# Option C: Look in the exact same directory (Matches: project/house_model.pkl)
path_option_c = os.path.join(SCRIPT_DIR, 'house_model.pkl')

# Determine which path actually exists
if os.path.exists(path_option_a):
    model_path = path_option_a
elif os.path.exists(path_option_b):
    model_path = path_option_b
else:
    model_path = path_option_c

# Attempt to load the model
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(f"❌ Model file not found! We searched in vain at: `{model_path}`. Please verify your folder structure.")
    st.stop()
# -----------------------------------

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
    # Create a DataFrame matching the model's expected feature names
    input_features = pd.DataFrame([{
        'SquareFeet': sqft,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'YearBuilt': year_built,
        'Neighborhood_Score': location_score
    }])
    
    # Generate prediction
    predicted_price = model.predict(input_features)[0]
    st.success(f"💰 Estimated Market Value: **${predicted_price:,.2f}**")
