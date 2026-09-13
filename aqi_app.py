import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pakistan AQI Predictor", page_icon="🌫️", layout="wide")

st.title("🌫️ Pakistan AQI Predictor")
st.caption("Predicts Air Quality Index value and health category using a tuned Gradient Boosting model")

# Load model, encoders, and city lookup
@st.cache_resource
def load_artifacts():
    reg_model = joblib.load("aqi_model.pkl")
    clf_model = joblib.load("aqi_classifier.pkl")
    le_city = joblib.load("le_city.pkl")
    le_season = joblib.load("le_season.pkl")
    le_province = joblib.load("le_province.pkl")
    with open("city_lookup.json") as f:
        city_lookup = json.load(f)
    return reg_model, clf_model, le_city, le_season, le_province, city_lookup

reg_model, clf_model, le_city, le_season, le_province, city_lookup = load_artifacts()

SEASONS = list(le_season.classes_)
CITIES = list(le_city.classes_)

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Inputs")
    city = st.selectbox("City", CITIES)
    month = st.slider("Month", 1, 12, 1)
    season = st.selectbox("Season", SEASONS)
    is_smog_season = st.checkbox("Smog season active?", value=(season == "Winter"))
    is_crop_burning_season = st.checkbox("Crop-burning season active?")
    is_monsoon_season = st.checkbox("Monsoon season active?")

    predict_btn = st.button("Predict AQI", type="primary", use_container_width=True)

with col2:
    if predict_btn:
        meta = city_lookup[city]

        row = pd.DataFrame([{
            "month": month,
            "city_encoded": le_city.transform([city])[0],
            "season_encoded": le_season.transform([season])[0],
            "province_encoded": le_province.transform([meta["province"]])[0],
            "is_smog_season": int(is_smog_season),
            "is_crop_burning_season": int(is_crop_burning_season),
            "is_monsoon_season": int(is_monsoon_season),
            "is_industrial_hub": meta["is_industrial_hub"],
            "is_coastal": meta["is_coastal"],
            "is_capital": meta["is_capital"],
            "population_millions": meta["population_millions"]
        }])

        pred_aqi = reg_model.predict(row)[0]
        pred_category = clf_model.predict(row)[0]

        st.subheader("Prediction")
        m1, m2 = st.columns(2)
        m1.metric("Predicted AQI Value", f"{pred_aqi:.1f}")
        m2.metric("Predicted Health Category", pred_category)

        st.divider()
        st.subheader("What drove this prediction?")

        importances = reg_model.feature_importances_
        feat_names = row.columns.tolist()
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(imp_df["feature"], imp_df["importance"], color="#2E86AB")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance (Gradient Boosting model)")
        st.pyplot(fig)
    else:
        st.info("Set your inputs on the left and click **Predict AQI** to see results.")

st.divider()
st.caption("Model: Hyperparameter-tuned Gradient Boosting (R²=0.974 regression, 91.7% classification accuracy) — trained on Pakistan Air Quality Index dataset, 10 cities, 2015–2025.")
