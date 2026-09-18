import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from realtime_api import get_live_city_aqi

# Clean, professional browser tab title and modern icon
st.set_page_config(page_title="Pakistan AQI Intelligence Platform", page_icon="🟢", layout="wide")

# ==================================================================
#  DESIGN TOKENS — Enterprise Dark-Slate Theme
# ==================================================================
PRIMARY = "#0F172A"       # Deep slate navy
ACCENT_GREEN = "#10B981"  # Emerald accent
BORDER_COLOR = "#E2E8F0"  # Subtle structural gray
CARD_BG = "#FFFFFF"       # Crisp white card surfaces
TEXT_MAIN = "#0F172A"     # High-contrast primary text
TEXT_MUTED = "#64748B"    # Secondary text

NAV = [
    ("predict", "Predict",           "Single-city real-time feed & predictive engine"),
    ("map",     "Spatial Map",       "Country-wide geographical pollution scoring"),
    ("trend",   "Historical Trend",  "Longitudinal 2015–2025 seasonal analytics"),
    ("compare", "Multi-Compare",     "Side-by-side multi-city scenario modeling"),
    ("heatmap", "Smog Heatmap",      "12-month seasonal intensity matrix"),
]
NAV_KEYS = [k for k, _, _ in NAV]

try:
    active = st.query_params.get("tab", "predict")
except Exception:
    active = st.experimental_get_query_params().get("tab", ["predict"])[0]
if active not in NAV_KEYS:
    active = "predict"

# ==================================================================
#  PROFESSIONAL CSS OVERHAUL
# ==================================================================
st.markdown(f"""
<style>
    .stApp {{
        background-color: #F8FAFC;
    }}
    h1, h2, h3, h4, p, label, .stMarkdown, .stCaption, li {{ 
        color: {TEXT_MAIN} !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    .stTabs {{ display: none !important; }}

    /* Masthead Header */
    .masthead-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 0;
        border-bottom: 1px solid {BORDER_COLOR};
        margin-bottom: 20px;
    }}
    .brand-title {{
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: {PRIMARY} !important;
        margin: 0;
    }}
    .brand-subtitle {{
        color: {TEXT_MUTED} !important;
        font-size: 0.92rem;
        margin-top: 2px;
    }}

    /* Segmented Navigation Bar */
    .nav-bar-container {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 6px;
        background: #E2E8F0;
        padding: 4px;
        border-radius: 12px;
        margin-bottom: 24px;
    }}
    .nav-item {{
        display: block;
        text-align: center;
        padding: 10px 0;
        font-size: 0.88rem;
        font-weight: 600;
        color: {TEXT_MUTED} !important;
        text-decoration: none !important;
        border-radius: 8px;
        transition: all 0.15s ease;
    }}
    .nav-item:hover {{
        color: {TEXT_MAIN} !important;
        background: rgba(255, 255, 255, 0.5);
    }}
    .nav-item.active {{
        background: #FFFFFF !important;
        color: {PRIMARY} !important;
        font-weight: 700;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}

    /* Hero Live Status Card */
    .hero-aqi-card {{
        border-radius: 16px;
        padding: 24px 28px;
        color: #FFFFFF !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
        margin-bottom: 24px;
    }}
    .hero-aqi-card * {{ color: #FFFFFF !important; }}
    .hero-card-top {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }}
    .hero-aqi-val {{
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1;
        letter-spacing: -0.02em;
    }}
    .hero-aqi-status {{
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 4px;
    }}
    .hero-card-bottom {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 20px;
        padding-top: 14px;
        border-top: 1px solid rgba(255,255,255,0.2);
        font-size: 0.88rem;
        font-weight: 500;
    }}

    /* Standard Cards */
    .content-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }}

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid {BORDER_COLOR};
    }}
    .sb-title {{
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {TEXT_MUTED} !important;
        margin-bottom: 16px;
    }}
</style>
""", unsafe_allow_html=True)

# ---------- Top Masthead Header ----------
st.markdown("""
<div class="masthead-container">
    <div>
        <h1 class="brand-title">Pakistan Air Quality Analytics</h1>
        <p class="brand-subtitle">Real-time Environmental Monitoring & Predictive Machine Learning Engine</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Navigation Segmented Control ----------
nav_html = "".join([
    f'<a class="nav-item {"active" if k == active else ""}" href="?tab={k}" target="_self">{label}</a>'
    for k, label, _ in NAV
])
st.markdown(f'<div class="nav-bar-container">{nav_html}</div>', unsafe_allow_html=True)

def style_axes(ax, fig):
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.tick_params(colors=TEXT_MAIN)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)

# ---------- Load Artifacts ----------
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
    "Good": "#10B981",
    "Moderate": "#F59E0B",
    "Unhealthy for Sensitive Groups": "#F97316",
    "Unhealthy": "#EF4444",
    "Very Unhealthy": "#8B5CF6",
    "Hazardous": "#6B21A8"
}

AQI_BANDS = [
    (0,   50,  "#10B981", "Good"),
    (50,  100, "#F59E0B", "Moderate"),
    (100, 150, "#F97316", "Unhealthy for Sensitive Groups"),
    (150, 200, "#EF4444", "Unhealthy"),
    (200, 300, "#8B5CF6", "Very Unhealthy"),
    (300, 500, "#6B21A8", "Hazardous"),
]

HEALTH_ADVICE = {
    "Good": "Air quality is satisfactory. Enjoy standard outdoor activities.",
    "Moderate": "Acceptable air quality. Sensitive individuals should consider reducing prolonged exertion.",
    "Unhealthy for Sensitive Groups": "Children, elderly, and individuals with respiratory issues should limit outdoor exertion.",
    "Unhealthy": "General public may begin to feel health effects. Avoid prolonged outdoor exposure.",
    "Very Unhealthy": "Health alert. Avoid outdoor activities and utilize indoor air filtration.",
    "Hazardous": "Emergency conditions. Stay indoors, keep windows closed, and wear an N95 mask if outdoors.",
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

def get_aqi_theme(val):
    if val <= 50:
        return "#10B981", "Good", "😊"
    elif val <= 100:
        return "#F59E0B", "Moderate", "😐"
    elif val <= 150:
        return "#F97316", "Unhealthy for Sensitive Groups", "😷"
    elif val <= 200:
        return "#EF4444", "Unhealthy", "😷"
    elif val <= 300:
        return "#8B5CF6", "Very Unhealthy", "🤢"
    else:
        return "#6B21A8", "Hazardous", "🚨"

def draw_gauge(value, color):
    fig, ax = plt.subplots(figsize=(5.5, 3.1), subplot_kw={"aspect": "equal"})
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    for lo, hi, band_color, _ in AQI_BANDS:
        theta1 = 180 - (lo / 500) * 180
        theta2 = 180 - (hi / 500) * 180
        wedge = plt.matplotlib.patches.Wedge(
            (0, 0), 1.0, theta2, theta1, width=0.32, facecolor=band_color, edgecolor="none"
        )
        ax.add_patch(wedge)

    clipped = max(0, min(value, 500))
    needle_theta = np.radians(180 - (clipped / 500) * 180)
    ax.plot([0, 0.82 * np.cos(needle_theta)], [0, 0.82 * np.sin(needle_theta)],
             color=PRIMARY, linewidth=3, solid_capstyle="round")
    ax.add_patch(plt.matplotlib.patches.Circle((0, 0), 0.05, color=PRIMARY))

    ax.text(0, -0.28, f"{value:.0f}", ha="center", va="center",
             fontsize=26, fontweight="bold", color=color)
    ax.text(0, -0.5, "AQI", ha="center", va="center", fontsize=11, color=TEXT_MUTED)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.55, 1.1)
    ax.axis("off")
    return fig

# ==================================================================
#  SIDEBAR
# ==================================================================
with st.sidebar:
    if active == "predict":
        st.markdown('<p class="sb-title">Configuration</p>', unsafe_allow_html=True)
        city = st.selectbox("City", CITIES, key="p_city")
        month = month_selector("Month", key="p_month")
        season = st.selectbox("Season", SEASONS, key="p_season")
        is_smog_season = st.toggle("Smog season active", value=(season == "Winter"), key="p_smog")
        is_crop_burning_season = st.toggle("Crop-burning season active", key="p_crop")
        is_monsoon_season = st.toggle("Monsoon season active", key="p_monsoon")
        predict_btn = st.button("Generate Prediction", type="primary", use_container_width=True)

    elif active == "map":
        st.markdown('<p class="sb-title">Spatial Inputs</p>', unsafe_allow_html=True)
        map_month = month_selector("Month", key="m_month")
        map_season = st.selectbox("Season", SEASONS, key="m_season")
        map_smog = st.toggle("Smog season active", value=(map_season == "Winter"), key="m_smog")
        map_crop = st.toggle("Crop-burning season active", key="m_crop")
        map_monsoon = st.toggle("Monsoon season active", key="m_monsoon")

    elif active == "trend":
        st.markdown('<p class="sb-title">Analytics Filters</p>', unsafe_allow_html=True)
        trend_city = st.selectbox("Select City", CITIES, key="t_city")
        show_prediction = st.toggle("Overlay current prediction", value=True, key="t_overlay")

    elif active == "compare":
        st.markdown('<p class="sb-title">Comparison Settings</p>', unsafe_allow_html=True)
        compare_cities = st.multiselect("Select Cities (2-4)", CITIES, default=CITIES[:2], max_selections=4, key="c_cities")
        cmp_month = month_selector("Month", key="c_month")
        cmp_season = st.selectbox("Season", SEASONS, key="c_season")
        cmp_smog = st.toggle("Smog season active", value=(cmp_season == "Winter"), key="c_smog")

    elif active == "heatmap":
        st.markdown('<p class="sb-title">Matrix Controls</p>', unsafe_allow_html=True)
        hm_season = st.selectbox("Season", SEASONS, key="h_season")
        hm_smog = st.toggle("Smog season active", value=(hm_season == "Winter"), key="h_smog")
        hm_crop = st.toggle("Crop-burning season active", key="h_crop")
        hm_monsoon = st.toggle("Monsoon season active", key="h_monsoon")

# ===================== TAB 1: PREDICT =====================
if active == "predict":
    live_data = get_live_city_aqi(city, city_lookup)
    if live_data and live_data.get("live_aqi") is not None:
        aqi_val = live_data["live_aqi"]
        card_bg, cat_label, avatar = get_aqi_theme(aqi_val)

        st.markdown(f"""
        <div class="hero-aqi-card" style="background: {card_bg};">
            <div class="hero-card-top">
                <div>
                    <span style="font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; opacity: 0.85; text-transform: uppercase;">LIVE MONITORING FEED — {city.upper()}</span>
                    <div class="hero-aqi-val">{aqi_val} <span style="font-size: 1rem; font-weight: 600;">US AQI</span></div>
                    <div class="hero-aqi-status">{cat_label}</div>
                </div>
                <div style="font-size: 3.2rem;">{avatar}</div>
            </div>
            <div class="hero-card-bottom">
                <span>📍 Station: {live_data.get('station', city)}</span>
                <span>Main Pollutant: PM2.5 ({live_data.get('pm25', 'N/A')} µg/m³)</span>
                <span>Updated: {live_data.get('time', 'N/A')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if predict_btn:
        row = build_row(city, month, season, is_smog_season, is_crop_burning_season, is_monsoon_season)
        pred_aqi = reg_model.predict(row)[0]
        pred_category = clf_model.predict(row)[0]
        cat_color = CATEGORY_COLORS.get(pred_category, PRIMARY)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(draw_gauge(pred_aqi, cat_color))
        with col2:
            st.metric("Predicted AQI Value", f"{pred_aqi:.1f}")
            st.metric("Health Classification", pred_category)

        st.markdown(
            f"""<div class="content-card" style="border-left: 4px solid {cat_color};">
            <p style="font-weight: 700; margin: 0 0 4px 0;">Health Recommendation — {pred_category}</p>
            <p style="margin: 0; color: {TEXT_MUTED};">{HEALTH_ADVICE.get(pred_category, "")}</p>
            </div>""",
            unsafe_allow_html=True,
        )

# ===================== TAB 2: MAP =====================
elif active == "map":
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
    st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), size=8000)
    st.dataframe(map_df.sort_values("Predicted AQI", ascending=False), use_container_width=True, hide_index=True)

# ===================== TAB 3: TREND =====================
elif active == "trend":
    city_hist = history[history["city"] == trend_city].copy()
    city_hist["date"] = pd.to_datetime(city_hist["year"].astype(str) + "-" + city_hist["month"].astype(str) + "-01")
    city_hist = city_hist.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    style_axes(ax, fig)
    ax.plot(city_hist["date"], city_hist["aqi_us"], color=PRIMARY, linewidth=1.5)
    ax.set_ylabel("AQI", color=TEXT_MAIN)
    st.pyplot(fig)

# ===================== TAB 4: COMPARE =====================
elif active == "compare":
    if len(compare_cities) >= 2:
        cmp_rows = []
        for c in compare_cities:
            row = build_row(c, cmp_month, cmp_season, cmp_smog, False, False)
            pred = reg_model.predict(row)[0]
            cat = clf_model.predict(row)[0]
            cmp_rows.append({"City": c, "Predicted AQI": round(pred, 1), "Category": cat})

        cmp_df = pd.DataFrame(cmp_rows)
        fig, ax = plt.subplots(figsize=(8, 3.5))
        style_axes(ax, fig)
        ax.bar(cmp_df["City"], cmp_df["Predicted AQI"], color=PRIMARY)
        st.pyplot(fig)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

# ===================== TAB 5: HEATMAP =====================
elif active == "heatmap":
    grid = np.zeros((len(CITIES), 12))
    for i, c in enumerate(CITIES):
        for m in range(1, 13):
            row = build_row(c, m, hm_season, hm_smog, hm_crop, hm_monsoon)
            grid[i, m - 1] = reg_model.predict(row)[0]

    fig, ax = plt.subplots(figsize=(11, 5))
    style_axes(ax, fig)
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels([m[:3] for m in MONTH_NAMES])
    ax.set_yticks(range(len(CITIES)))
    ax.set_yticklabels(CITIES)
    st.pyplot(fig)
