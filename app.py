import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pakistan AQI Predictor", layout="wide")

# ==================================================================
#  THEME TOKENS — one accent colour per section, used everywhere
# ==================================================================
ACCENT = {
    "predict": "#22D3EE",   # cyan
    "map":     "#34D399",   # green
    "trend":   "#A78BFA",   # violet
    "compare": "#FBBF24",   # amber
}
TITLE_GOLD = "#FFC94A"      # warm title against the cool background
INK = "#E8F1F8"             # body text
MUTED = "#93AEC2"           # secondary text

# ---------- Custom styling ----------
st.markdown(f"""
<style>
    /* ---------- base surface ---------- */
    .stApp {{
        background: linear-gradient(165deg, #0B1A21 0%, #14303B 45%, #1D4655 100%);
        background-attachment: fixed;
    }}
    /* The whole page picks up a faint wash of the active tab's colour.
       Browsers without :has() simply keep the base gradient. */
    .stApp:has([data-baseweb="tab-list"] > button:nth-child(1)[aria-selected="true"]) {{
        background: linear-gradient(165deg, #0B1A21 0%, #12313C 45%, #164A5C 100%);
    }}
    .stApp:has([data-baseweb="tab-list"] > button:nth-child(2)[aria-selected="true"]) {{
        background: linear-gradient(165deg, #0B1E1A 0%, #123A31 45%, #17544A 100%);
    }}
    .stApp:has([data-baseweb="tab-list"] > button:nth-child(3)[aria-selected="true"]) {{
        background: linear-gradient(165deg, #150F26 0%, #2A2050 45%, #3A2C6B 100%);
    }}
    .stApp:has([data-baseweb="tab-list"] > button:nth-child(4)[aria-selected="true"]) {{
        background: linear-gradient(165deg, #201705 0%, #3D2D0C 45%, #55401A 100%);
    }}

    h1, h2, h3, h4, p, label, .stMarkdown, .stCaption, li {{
        color: {INK} !important;
    }}

    /* ---------- masthead ---------- */
    .masthead {{
        padding: 26px 0 18px 0;
        margin-bottom: 4px;
    }}
    .masthead .wordmark {{
        font-size: 3.1rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: {TITLE_GOLD} !important;
        text-shadow: 0 0 28px rgba(255, 201, 74, 0.35);
        margin: 0;
    }}
    .masthead .titleblock {{ display: inline-block; }}
    .masthead .goldrule {{
        display: block;
        height: 5px;
        width: 100%;
        margin: 14px 0 14px 0;
        border-radius: 999px;
        background: {TITLE_GOLD};
    }}
    .masthead .standfirst {{
        color: {MUTED} !important;
        font-size: 1.02rem;
        max-width: 68ch;
        margin: 0;
    }}

    /* ---------- tab bar: four separate, differently coloured pill buttons ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 14px;
        border-bottom: none !important;
        padding: 8px 0 18px 0;
        flex-wrap: wrap;
    }}
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
        background-color: transparent !important;
    }}
    .stTabs button[role="tab"] {{
        border-radius: 999px !important;
        padding: 10px 24px !important;
        height: auto !important;
        font-weight: 800 !important;
        border: 2px solid transparent !important;
        background-color: rgba(255,255,255,0.06);
        transition: transform .15s ease, box-shadow .15s ease;
    }}
    .stTabs button[role="tab"] * {{
        color: inherit !important;
        font-weight: 800 !important;
    }}
    .stTabs button[role="tab"]:hover {{ transform: translateY(-2px); }}
    .stTabs button[role="tab"]:focus-visible {{ outline: 3px solid {TITLE_GOLD}; outline-offset: 3px; }}

    /* --- 1. Predict (cyan) --- */
    .stTabs button[role="tab"]:nth-child(1) {{
        color: {ACCENT['predict']} !important;
        background-color: {ACCENT['predict']}1F !important;
        border-color: {ACCENT['predict']}99 !important;
    }}
    .stTabs button[role="tab"]:nth-child(1)[aria-selected="true"] {{
        color: #04161B !important;
        background-color: {ACCENT['predict']} !important;
        border-color: {ACCENT['predict']} !important;
        box-shadow: 0 6px 20px {ACCENT['predict']}66;
    }}
    /* --- 2. Map (green) --- */
    .stTabs button[role="tab"]:nth-child(2) {{
        color: {ACCENT['map']} !important;
        background-color: {ACCENT['map']}1F !important;
        border-color: {ACCENT['map']}99 !important;
    }}
    .stTabs button[role="tab"]:nth-child(2)[aria-selected="true"] {{
        color: #04160F !important;
        background-color: {ACCENT['map']} !important;
        border-color: {ACCENT['map']} !important;
        box-shadow: 0 6px 20px {ACCENT['map']}66;
    }}
    /* --- 3. Historical Trend (violet) --- */
    .stTabs button[role="tab"]:nth-child(3) {{
        color: {ACCENT['trend']} !important;
        background-color: {ACCENT['trend']}26 !important;
        border-color: {ACCENT['trend']}99 !important;
    }}
    .stTabs button[role="tab"]:nth-child(3)[aria-selected="true"] {{
        color: #120A24 !important;
        background-color: {ACCENT['trend']} !important;
        border-color: {ACCENT['trend']} !important;
        box-shadow: 0 6px 20px {ACCENT['trend']}66;
    }}
    /* --- 4. Compare Cities (amber) --- */
    .stTabs button[role="tab"]:nth-child(4) {{
        color: {ACCENT['compare']} !important;
        background-color: {ACCENT['compare']}26 !important;
        border-color: {ACCENT['compare']}99 !important;
    }}
    .stTabs button[role="tab"]:nth-child(4)[aria-selected="true"] {{
        color: #1C1302 !important;
        background-color: {ACCENT['compare']} !important;
        border-color: {ACCENT['compare']} !important;
        box-shadow: 0 6px 20px {ACCENT['compare']}66;
    }}

    /* ---------- section banner that opens each tab ---------- */
    .panel-head {{
        border-radius: 0 14px 14px 0;
        padding: 16px 20px;
        margin: 22px 0 6px 0;
    }}
    .panel-head .panel-title {{
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.01em;
    }}
    .panel-head .panel-sub {{
        color: {MUTED} !important;
        font-size: 0.95rem;
        margin: 4px 0 0 0;
    }}
    .rule {{ height: 2px; border: 0; margin: 0 0 20px 0; border-radius: 2px; }}

    /* ---------- metrics ---------- */
    div[data-testid="stMetric"] {{
        background-color: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 14px;
        padding: 16px;
    }}
    div[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-weight: 600; }}
    div[data-testid="stMetricValue"] {{ color: #FFFFFF !important; }}

    /* ---------- buttons ---------- */
    .stButton>button {{
        background: linear-gradient(90deg, {ACCENT['predict']}, #0072FF);
        color: #041318;
        border-radius: 10px;
        font-weight: 800;
        border: none;
        padding: 12px 0px;
        transition: transform 0.15s ease;
    }}
    .stButton>button:hover {{ transform: translateY(-1px); }}
    .stButton>button:focus-visible {{ outline: 3px solid {TITLE_GOLD}; outline-offset: 2px; }}

    /* ---------- inputs: keep them dark instead of white ---------- */
    div[data-baseweb="select"] > div {{
        background-color: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 10px;
        color: {INK} !important;
    }}
    div[data-baseweb="select"] div, div[data-baseweb="select"] span {{ color: {INK} !important; }}
    div[data-baseweb="popover"] li {{ color: #0B1A21 !important; }}
    [data-testid="stDataFrame"] {{ border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)

# ---------- Masthead ----------
st.markdown("""
<div class="masthead">
  <div class="titleblock">
    <p class="wordmark">🌫️ Pakistan AQI Predictor</p>
    <span class="goldrule"></span>
  </div>
  <p class="standfirst">Predicts the Air Quality Index value and health category for 10 Pakistani cities
  using a tuned Gradient Boosting model trained on 2015–2025 data.</p>
</div>
""", unsafe_allow_html=True)


def panel(key, title, subtitle):
    """Opens a tab with a colour-coded banner and a matching rule."""
    c = ACCENT[key]
    st.markdown(
        f"""
        <div class="panel-head" style="border-left:6px solid {c};
             background:linear-gradient(90deg, {c}26 0%, rgba(255,255,255,0.02) 85%);">
          <p class="panel-title" style="color:{c} !important;">{title}</p>
          <p class="panel-sub">{subtitle}</p>
        </div>
        <hr class="rule" style="background:{c};">
        """,
        unsafe_allow_html=True,
    )


def style_axes(ax, fig, accent):
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(accent)
        spine.set_alpha(0.6)


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
    "Good": "#2ECC71",
    "Moderate": "#F1C40F",
    "Unhealthy": "#E67E22",
    "Unhealthy for Sensitive Groups": "#E67E22",
    "Very Unhealthy": "#C0392B",
    "Hazardous": "#7D3C98"
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
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔮 Predict", "🗺️ Map (All Cities)", "📈 Historical Trend", "⚖️ Compare Cities"]
)

# ===== TAB 1: Predict =====
with tab1:
    panel("predict", "🔮 Predict", "Pick a city and month, then read the AQI value and health category.")
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
            cat_color = CATEGORY_COLORS.get(pred_category, ACCENT["predict"])

            st.subheader("Prediction")
            m1, m2 = st.columns(2)
            m1.metric("Predicted AQI value", f"{pred_aqi:.1f}")
            m2.metric("Health category", pred_category)
            st.markdown(
                f"<hr class='rule' style='background:{cat_color};margin-top:14px;'>",
                unsafe_allow_html=True,
            )

            st.subheader("What drove this prediction?")
            importances = reg_model.feature_importances_
            feat_names = row.columns.tolist()
            imp_df = pd.DataFrame(
                {"feature": feat_names, "importance": importances}
            ).sort_values("importance", ascending=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            style_axes(ax, fig, ACCENT["predict"])
            ax.barh(imp_df["feature"], imp_df["importance"], color=ACCENT["predict"])
            ax.set_xlabel("Importance", color=INK)
            ax.set_title("Feature importance (Gradient Boosting model)", color=ACCENT["predict"])
            st.pyplot(fig)
        else:
            st.info("Set your inputs on the left, then click **Predict AQI**.")

# ===== TAB 2: Map =====
with tab2:
    panel("map", "🗺️ Map (All Cities)", "Every city scored for the same month and season, mapped and ranked.")
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
        st.map(
            map_df.rename(columns={"lat": "latitude", "lon": "longitude"}),
            size=8000,
            color=ACCENT["map"],
        )
    else:
        st.warning("Some cities have no latitude or longitude in city_lookup.json, so only the table is shown.")

    st.dataframe(
        map_df.sort_values("Predicted AQI", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# ===== TAB 3: Historical Trend =====
with tab3:
    panel("trend", "📈 Historical Trend", "Recorded AQI for one city, month by month, 2015 to 2025.")
    trend_city = st.selectbox("Select a city", CITIES, key="t_city")
    city_hist = history[history["city"] == trend_city].copy()
    city_hist["date"] = pd.to_datetime(
        city_hist["year"].astype(str) + "-" + city_hist["month"].astype(str) + "-01"
    )
    city_hist = city_hist.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 4))
    style_axes(ax, fig, ACCENT["trend"])
    ax.plot(city_hist["date"], city_hist["aqi_us"], color=ACCENT["trend"], linewidth=1.6)
    ax.fill_between(city_hist["date"], city_hist["aqi_us"], color=ACCENT["trend"], alpha=0.15)
    ax.set_ylabel("AQI", color=INK)
    ax.set_title(f"AQI trend: {trend_city} (2015–2025)", color=ACCENT["trend"])
    ax.grid(alpha=0.15, color=INK)
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Average AQI", f"{city_hist['aqi_us'].mean():.1f}")
    c2.metric("Peak AQI", f"{city_hist['aqi_us'].max():.1f}")
    c3.metric("Lowest AQI", f"{city_hist['aqi_us'].min():.1f}")

# ===== TAB 4: Compare Cities =====
with tab4:
    panel("compare", "⚖️ Compare Cities", "Put two to four cities side by side under identical conditions.")
    compare_cities = st.multiselect(
        "Select 2–4 cities", CITIES, default=CITIES[:2], max_selections=4, key="c_cities"
    )
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
        style_axes(ax, fig, ACCENT["compare"])
        colors = [CATEGORY_COLORS.get(cat, ACCENT["compare"]) for cat in cmp_df["Category"]]
        bars = ax.bar(cmp_df["City"], cmp_df["Predicted AQI"], color=colors)
        ax.bar_label(bars, fmt="%.0f", color=INK, padding=3)
        ax.set_ylabel("Predicted AQI", color=INK)
        ax.set_title("City comparison", color=ACCENT["compare"])
        st.pyplot(fig)

        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
    else:
        st.info("Pick at least 2 cities to compare.")

st.divider()
st.caption(
    "Model: hyperparameter-tuned Gradient Boosting (R² = 0.974 regression, 91.7% classification accuracy) — "
    "Pakistan Air Quality Index dataset, 10 cities, 2015–2025."
)
