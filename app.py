import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction Tool")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "Model", "house_model.joblib")
METRICS_PATH = os.path.join(SCRIPT_DIR, "Model", "metrics.json")
IMPORTANCE_PATH = os.path.join(SCRIPT_DIR, "Model", "feature_importance.json")
FEATURES = ["SquareFeet", "Bedrooms", "Bathrooms", "HouseAge", "Neighborhood_Score"]


@st.cache_resource
def load_model():
    """Cached so the model is loaded once per session, not on every click."""
    with open(MODEL_PATH, "rb") as f:
        return joblib.load(f)


@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"❌ Model file not found at `{MODEL_PATH}`. "
        "Run `python Model/train_model.py` first to train and save it."
    )
    st.stop()

metrics = load_json(METRICS_PATH)
importances = load_json(IMPORTANCE_PATH)

tab_single, tab_batch = st.tabs(["🔢 Single Prediction", "📄 Batch (CSV) Prediction"])

# ---------- Single prediction ----------
with tab_single:
    st.header("Property Details")
    col1, col2 = st.columns(2)
    with col1:
        sqft = st.number_input("Total Square Footage", min_value=300, max_value=10000, value=1500)
        bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=8, value=3)
        bathrooms = st.slider("Number of Bathrooms", min_value=1.0, max_value=6.0, value=2.0, step=0.5)
    with col2:
        year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2010)
        location_score = st.slider("Neighborhood Rating (1-10)", min_value=1.0, max_value=10.0, value=5.0)

    if bathrooms > bedrooms + 2:
        st.warning("That's a lot of bathrooms relative to bedrooms — double check the input.")

    if st.button("Calculate Estimated Value", type="primary"):
        house_age = 2026 - year_built
        input_features = pd.DataFrame([{
            "SquareFeet": sqft,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "HouseAge": house_age,
            "Neighborhood_Score": location_score,
        }])[FEATURES]

        predicted_price = model.predict(input_features)[0]
        price_per_sqft = predicted_price / sqft

        st.success(f"💰 Estimated Market Value: **${predicted_price:,.2f}**")
        st.caption(f"≈ ${price_per_sqft:,.2f} per square foot")

        if metrics:
            mae = metrics["all_results"][metrics["best_model"]]["mae"]
            st.info(
                f"Model: **{metrics['best_model']}** · typical error on test data: "
                f"± ${mae:,.0f} (so treat this as a range, not an exact figure)."
            )

    if importances:
        with st.expander("📊 What drives this prediction most?"):
            imp_df = pd.DataFrame(
                sorted(importances.items(), key=lambda x: -x[1]),
                columns=["Feature", "Importance"],
            )
            st.bar_chart(imp_df.set_index("Feature"))

    if metrics:
        with st.expander("🧪 Model accuracy details"):
            st.json(metrics["all_results"])

# ---------- Batch prediction ----------
with tab_batch:
    st.write(
        "Upload a CSV with columns: `SquareFeet, Bedrooms, Bathrooms, YearBuilt, "
        "Neighborhood_Score` to get predictions for many houses at once."
    )
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            required_cols = {"SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt", "Neighborhood_Score"}
            missing = required_cols - set(batch_df.columns)
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                batch_df["HouseAge"] = 2026 - batch_df["YearBuilt"]
                batch_df["Predicted_Price"] = model.predict(batch_df[FEATURES])
                st.dataframe(batch_df)
                st.download_button(
                    "Download predictions as CSV",
                    batch_df.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Couldn't process that file: {e}")
