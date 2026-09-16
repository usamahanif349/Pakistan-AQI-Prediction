import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pakistan AQI Predictor", page_icon="🌫️", layout="wide")

# ---------- Custom Styling ----------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1 { color: #f0f2f6; font-weight: 800; }
    div[data-testid="stMetric"] {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
    }
    div[data-testid="stMetricValue"] { color: #2E86AB; }
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 10px 0px;
    }
    .stButton>button:hover { background-color: #1e5f7a; }
</style>
""", unsafe_allow_html=True)

st.title("🌫️ Pakistan AQI Predictor")
st.caption("Predicts Air Quality Index value and health category using a tuned Gradient Boosting model — trained on 10 Pakistani cities, 2015–2025")

# ---------- Load artifacts ----------
@st.cache_resource
def load_artifacts():
    reg_model = joblib.load("aqi_model.pkl")
    clf_model = joblib.load("aqi_classifier.pkl")
    le_city = joblib.load("le_city.pkl")
    le_season = joblib.load("le_season.pkl")
    le_province = joblib.load("le_province.pkl")
    with open("city_lookup.json") as f:
        city_lookup = json.load(f)
    history = pd.read_csv("city_history.csv")
    return reg_model, clf_model, le_city, le_season, le_province, city_lookup, history

reg_model, clf_model, le_city, le_season, le_province, city_lookup, history = load_artifacts()

SEASONS = list(le_season.classes_)
CITIES = list(le_city.classes_)

CATEGORY_COLORS = {
    "Moderate": "#F1C40F",
    "Unhealthy": "#E67E22",
    "Unhealthy for Sensitive Groups": "#E67E22",
    "Very Unhealthy": "#C0392B"
}

def build_row(city, month, season, is_smog, is_crop, is_monsoon):
    meta = city_lookup[city]
    return pd.DataFrame([{
        "month": month,
        "city_encoded": le_city.transform([city])[0],
        "season_encoded": le_season.transform([season])[0],
        "province_encoded": le_province.transform([meta["province"]])[0],
        "is_smog_season": int(is_smog),
        "is_crop_burning_season": int(is_crop),
        "is_monsoon_season": int(is_monsoon),
        "is_industrial_hub": meta["is_industrial_hub"],
        "is_coastal": meta["is_coastal"],
        "is_capital": meta["is_capital"],
        "population_millions": meta["population_millions"]
    }])

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "🗺️ Map (All Cities)", "📈 Historical Trend", "⚖️ Compare Cities"])

# ===== TAB 1: Predict =====
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Inputs")
        city = st.selectbox("City", CITIES, key="p_city")
        month = st.slider("Month", 1, 12, 6, key="p_month")
        season = st.selectbox("Season", SEASONS, key="p_season")
        is_smog_season = st.checkbox("Smog season active?", value=(season == "Winter"), key="p_smog")
        is_crop_burning_season = st.checkbox("Crop-burning season active?", key="p_crop")
        is_monsoon_season = st.checkbox("Monsoon season active?", key="p_monsoon")
        predict_btn = st.button("Predict AQI", type="primary", use_container_width=True)

    with col2:
        if predict_btn:
            row = build_row(city, month, season, is_smog_season, is_crop_burning_season, is_monsoon_season)
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

# ===== TAB 2: Map =====
with tab2:
    st.subheader("Predicted AQI Across All Cities")
    map_month = st.slider("Month", 1, 12, 6, key="m_month")
    map_season = st.selectbox("Season", SEASONS, key="m_season")
    map_smog = st.checkbox("Smog season active?", value=(map_season == "Winter"), key="m_smog")
    map_crop = st.checkbox("Crop-burning season active?", key="m_crop")
    map_monsoon = st.checkbox("Monsoon season active?", key="m_monsoon")

    map_rows = []
    for c in CITIES:
        row = build_row(c, map_month, map_season, map_smog, map_crop, map_monsoon)
        pred = reg_model.predict(row)[0]
        cat = clf_model.predict(row)[0]
        meta = city_lookup[c]
        map_rows.append({
            "city": c, "lat": meta.get("latitude"), "lon": meta.get("longitude"),
            "Predicted AQI": round(pred, 1), "Category": cat
        })

    map_df = pd.DataFrame(map_rows)

    if map_df["lat"].notna().all():
        st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), size=200000, color="#E6550D")
    else:
        st.warning("Latitude/longitude not found for some cities — showing table only.")

    st.dataframe(map_df.sort_values("Predicted AQI", ascending=False), use_container_width=True, hide_index=True)

# ===== TAB 3: Historical Trend =====
with tab3:
    st.subheader("Historical AQI Trend")
    trend_city = st.selectbox("Select a city", CITIES, key="t_city")
    city_hist = history[history["city"] == trend_city].copy()
    city_hist["date"] = pd.to_datetime(city_hist["year"].astype(str) + "-" + city_hist["month"].astype(str) + "-01")
    city_hist = city_hist.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(city_hist["date"], city_hist["aqi_us"], color="#2E86AB", linewidth=1.2)
    ax.set_ylabel("AQI")
    ax.set_title(f"AQI Trend: {trend_city} (2015–2025)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Average AQI", f"{city_hist['aqi_us'].mean():.1f}")
    c2.metric("Peak AQI", f"{city_hist['aqi_us'].max():.1f}")
    c3.metric("Lowest AQI", f"{city_hist['aqi_us'].min():.1f}")

# ===== TAB 4: Compare Cities =====
with tab4:
    st.subheader("Compare Predicted AQI Across Cities")
    compare_cities = st.multiselect("Select 2–4 cities", CITIES, default=CITIES[:2], max_selections=4, key="c_cities")
    cmp_month = st.slider("Month", 1, 12, 6, key="c_month")
    cmp_season = st.selectbox("Season", SEASONS, key="c_season")
    cmp_smog = st.checkbox("Smog season active?", value=(cmp_season == "Winter"), key="c_smog")

    if len(compare_cities) >= 2:
        cmp_rows = []
        for c in compare_cities:
            row = build_row(c, cmp_month, cmp_season, cmp_smog, False, False)
            pred = reg_model.predict(row)[0]
            cat = clf_model.predict(row)[0]
            cmp_rows.append({"City": c, "Predicted AQI": round(pred, 1), "Category": cat})

        cmp_df = pd.DataFrame(cmp_rows)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [CATEGORY_COLORS.get(cat, "#2E86AB") for cat in cmp_df["Category"]]
        ax.bar(cmp_df["City"], cmp_df["Predicted AQI"], color=colors)
        ax.set_ylabel("Predicted AQI")
        ax.set_title("City Comparison")
        st.pyplot(fig)

        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
    else:
        st.info("Select at least 2 cities to compare.")

st.divider()
st.caption("Model: Hyperparameter-tuned Gradient Boosting (R²=0.974 regression, 91.7% classification accuracy) — Pakistan Air Quality Index dataset, 10 cities, 2015–2025.")
