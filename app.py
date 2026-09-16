import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pakistan AQI Predictor", page_icon="🌫️", layout="wide")

# ---------- Custom Styling: weather-app style dark theme ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(160deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    h1, h2, h3, p, label, .stMarkdown, .stCaption {
        color: #eaf6ff !important;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 14px;
        padding: 16px;
    }
    div[data-testid="stMetricLabel"] p {
        color: #9fd8ff !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 10px;
        font-weight: 700;
        border: none;
        padding: 12px 0px;
        transition: transform 0.15s;
    }
    .stButton>button:hover { transform: scale(1.02); }
    div[data-testid="stSelectbox"] > div, div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        color: #eaf6ff;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.06);
        border-radius: 8px 8px 0px 0px;
        color: #eaf6ff;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 198, 255, 0.25) !important;
    }
    [data-testid="stDataFrame"] { border-radius: 10px; }
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
MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
MONTH_OPTIONS = [f"{i+1} — {name}" for i, name in enumerate(MONTH_NAMES)]

CATEGORY_COLORS = {
    "Moderate": "#F1C40F",
    "Unhealthy": "#E67E22",
    "Unhealthy for Sensitive Groups": "#E67E22",
    "Very Unhealthy": "#C0392B"
}

def month_selector(label, key, default_idx=5):
    choice = st.selectbox(label, MONTH_OPTIONS, index=default_idx, key=key)
    return int(choice.split(" — ")[0])

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
        month = month_selector("Month", key="p_month")
        season = st.selectbox("Season", SEASONS, key="p_season")
        is_smog_season = st.toggle("Smog season active?", value=(season == "Winter"), key="p_smog")
        is_crop_burning_season = st.toggle("Crop-burning season active?", key="p_crop")
        is_monsoon_season = st.toggle("Monsoon season active?", key="p_monsoon")
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
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            ax.barh(imp_df["feature"], imp_df["importance"], color="#00c6ff")
            ax.set_xlabel("Importance", color="#eaf6ff")
            ax.set_title("Feature Importance (Gradient Boosting model)", color="#eaf6ff")
            ax.tick_params(colors="#eaf6ff")
            for spine in ax.spines.values():
                spine.set_color("#eaf6ff")
            st.pyplot(fig)
        else:
            st.info("Set your inputs on the left and click **Predict AQI** to see results.")

# ===== TAB 2: Map =====
with tab2:
    st.subheader("Predicted AQI Across All Cities")
    c1, c2 = st.columns(2)
    with c1:
        map_month = month_selector("Month", key="m_month")
        map_season = st.selectbox("Season", SEASONS, key="m_season")
    with c2:
        map_smog = st.toggle("Smog season active?", value=(map_season == "Winter"), key="m_smog")
        map_crop = st.toggle("Crop-burning season active?", key="m_crop")
        map_monsoon = st.toggle("Monsoon season active?", key="m_monsoon")

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
        st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), size=8000, color="#00c6ff")
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
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.plot(city_hist["date"], city_hist["aqi_us"], color="#00c6ff", linewidth=1.4)
    ax.set_ylabel("AQI", color="#eaf6ff")
    ax.set_title(f"AQI Trend: {trend_city} (2015–2025)", color="#eaf6ff")
    ax.tick_params(colors="#eaf6ff")
    ax.grid(alpha=0.2, color="#eaf6ff")
    for spine in ax.spines.values():
        spine.set_color("#eaf6ff")
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Average AQI", f"{city_hist['aqi_us'].mean():.1f}")
    c2.metric("Peak AQI", f"{city_hist['aqi_us'].max():.1f}")
    c3.metric("Lowest AQI", f"{city_hist['aqi_us'].min():.1f}")

# ===== TAB 4: Compare Cities =====
with tab4:
    st.subheader("Compare Predicted AQI Across Cities")
    compare_cities = st.multiselect("Select 2–4 cities", CITIES, default=CITIES[:2], max_selections=4, key="c_cities")
    cmp_month = month_selector("Month", key="c_month")
    cmp_season = st.selectbox("Season", SEASONS, key="c_season")
    cmp_smog = st.toggle("Smog season active?", value=(cmp_season == "Winter"), key="c_smog")

    if len(compare_cities) >= 2:
        cmp_rows = []
        for c in compare_cities:
            row = build_row(c, cmp_month, cmp_season, cmp_smog, False, False)
            pred = reg_model.predict(row)[0]
            cat = clf_model.predict(row)[0]
            cmp_rows.append({"City": c, "Predicted AQI": round(pred, 1), "Category": cat})

        cmp_df = pd.DataFrame(cmp_rows)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        colors = [CATEGORY_COLORS.get(cat, "#00c6ff") for cat in cmp_df["Category"]]
        ax.bar(cmp_df["City"], cmp_df["Predicted AQI"], color=colors)
        ax.set_ylabel("Predicted AQI", color="#eaf6ff")
        ax.set_title("City Comparison", color="#eaf6ff")
        ax.tick_params(colors="#eaf6ff")
        for spine in ax.spines.values():
            spine.set_color("#eaf6ff")
        st.pyplot(fig)

        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
    else:
        st.info("Select at least 2 cities to compare.")

st.divider()
st.caption("Model: Hyperparameter-tuned Gradient Boosting (R²=0.974 regression, 91.7% classification accuracy) — Pakistan Air Quality Index dataset, 10 cities, 2015–2025.")
