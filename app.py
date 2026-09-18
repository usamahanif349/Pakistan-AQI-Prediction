import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from realtime_api import get_live_city_aqi

# Modern tab title and icon
st.set_page_config(page_title="Pakistan AQI Intelligence Platform", page_icon="🟢", layout="wide")

# ==================================================================
#  DESIGN TOKENS — Enterprise Theme with Dark Sidebar
# ==================================================================
PRIMARY = "#0F172A"       # Deep slate navy
BORDER_COLOR = "#CBD5E1"  # Structured gray border
CARD_BG = "#FFFFFF"       # White card surfaces
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
#  CSS OVERHAUL
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

    /* Header Masthead */
    .masthead-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 0;
        border-bottom: 2px solid {BORDER_COLOR};
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

    /* Distinct Navigation Bar Container */
    .nav-bar-container {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 8px;
        background: #CBD5E1;
        padding: 6px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }}
    .nav-item {{
        display: block;
        text-align: center;
        padding: 11px 0;
        font-size: 0.9rem;
        font-weight: 700;
        color: #334155 !important;
        text-decoration: none !important;
        border-radius: 8px;
        transition: all 0.15s ease;
    }}
    .nav-item:hover {{
        color: {PRIMARY} !important;
        background: rgba(255, 255, 255, 0.6);
    }}
    .nav-item.active {{
        background: #FFFFFF !important;
        color: #2563EB !important;
        font-weight: 800;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }}

    /* Dynamic Live Hero Card */
    .hero-aqi-card {{
        border-radius: 16px;
        padding: 24px 28px;
        color: #FFFFFF !important;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.12);
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
        border-top: 1px solid rgba(255,255,255,0.25);
        font-size: 0.88rem;
        font-weight: 500;
    }}

    /* Standard Cards */
    .infocard {{
        border-radius: 12px;
        padding: 16px 18px;
        margin: 12px 0;
        border: 1px solid {BORDER_COLOR};
        background: #FFFFFF;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }}
    .infocard .infocard-title {{
        font-weight: 800; font-size: 1rem; margin: 0 0 6px 0; color: {PRIMARY} !important;
    }}
    .infocard p {{ margin: 2px 0; }}
    .chip-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
    .chip {{
        font-size: .8rem; font-weight: 700; padding: 4px 12px;
        border-radius: 999px; background: #E2E8F0; color: {PRIMARY} !important; border: 1px solid {BORDER_COLOR};
    }}

    /* Metric Font Fix — Prevents Text Cutoff (...) */
    div[data-testid="stMetric"] {{
        background-color: #FFFFFF;
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }}
    div[data-testid="stMetricValue"] {{ 
        color: {PRIMARY} !important; 
        font-size: 1.1rem !important; 
        line-height: 1.3 !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow: visible !important;
    }}

    /* Dark Contrast Sidebar Panel */
    section[data-testid="stSidebar"] {{
        background-color: #0F172A !important;
        border-right: 2px solid #1E293B;
    }}
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {{
        color: #F8FAFC !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {{
        color: #0F172A !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
    }}
    .sb-title {{
        font-size: 0.95rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8 !important;
        margin-bottom: 16px;
        padding-bottom: 6px;
        border-bottom: 2px solid #334155;
    }}
</style>
""", unsafe_allow_html=True)

# ---------- Top Masthead ----------
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
        st.markdown('<p class="sb-title">Predict Inputs</p>', unsafe_allow_html=True)
        city = st.selectbox("City", CITIES, key="p_city")
        month = month_selector("Month", key="p_month")
        season = st.selectbox("Season", SEASONS, key="p_season")
        is_smog_season = st.toggle("Smog season active?", value=(season == "Winter"), key="p_smog")
        is_crop_burning_season = st.toggle("Crop-burning season active?", key="p_crop")
        is_monsoon_season = st.toggle("Monsoon season active?", key="p_monsoon")
        predict_btn = st.button("Predict AQI", type="primary", use_container_width=True)

        meta = city_lookup[city]
        st.markdown(
            f"""
            <div class="infocard" style="background: #1E293B; border-color: #334155;">
              <p class="infocard-title" style="color: #F8FAFC !important;">{city}</p>
              <p style="color: #94A3B8 !important;">Province: {meta.get('province', '—')}</p>
              <p style="color: #94A3B8 !important;">Population: {meta.get('population_millions', '—')}M</p>
              <div class="chip-row">
                {'<span class="chip" style="background:#334155; color:#F8FAFC !important; border-color:#475569;">Industrial hub</span>' if meta.get('is_industrial_hub') else ''}
                {'<span class="chip" style="background:#334155; color:#F8FAFC !important; border-color:#475569;">Coastal</span>' if meta.get('is_coastal') else ''}
                {'<span class="chip" style="background:#334155; color:#F8FAFC !important; border-color:#475569;">Capital</span>' if meta.get('is_capital') else ''}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif active == "map":
        st.markdown('<p class="sb-title">Map Inputs</p>', unsafe_allow_html=True)
        map_month = month_selector("Month", key="m_month")
        map_season = st.selectbox("Season", SEASONS, key="m_season")
        map_smog = st.toggle("Smog season active?", value=(map_season == "Winter"), key="m_smog")
        map_crop = st.toggle("Crop-burning season active?", key="m_crop")
        map_monsoon = st.toggle("Monsoon season active?", key="m_monsoon")

    elif active == "trend":
        st.markdown('<p class="sb-title">Trend Inputs</p>', unsafe_allow_html=True)
        trend_city = st.selectbox("Select a city", CITIES, key="t_city")
        show_prediction = st.toggle("Overlay this month's prediction", value=True, key="t_overlay")

    elif active == "compare":
        st.markdown('<p class="sb-title">Compare Inputs</p>', unsafe_allow_html=True)
        compare_cities = st.multiselect("Select 2–4 cities", CITIES, default=CITIES[:2], max_selections=4, key="c_cities")
        cmp_month = month_selector("Month", key="c_month")
        cmp_season = st.selectbox("Season", SEASONS, key="c_season")
        cmp_smog = st.toggle("Smog season active?", value=(cmp_season == "Winter"), key="c_smog")

    elif active == "heatmap":
        st.markdown('<p class="sb-title">Heatmap Inputs</p>', unsafe_allow_html=True)
        hm_season = st.selectbox("Season", SEASONS, key="h_season")
        hm_smog = st.toggle("Smog season active?", value=(hm_season == "Winter"), key="h_smog")
        hm_crop = st.toggle("Crop-burning season active?", key="h_crop")
        hm_monsoon = st.toggle("Monsoon season active?", key="h_monsoon")

# ===================== PREDICT =====================
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

        st.subheader("Prediction")
        g1, g2 = st.columns([1, 1])
        with g1:
            st.pyplot(draw_gauge(pred_aqi, cat_color))
        with g2:
            st.metric("Predicted AQI value", f"{pred_aqi:.1f}")
            st.metric("Health category", pred_category)

            city_month_hist = history[(history["city"] == city) & (history["month"] == month)]["aqi_us"]
            if len(city_month_hist) >= 3:
                spread = city_month_hist.std()
                lo, hi = max(0, pred_aqi - spread), pred_aqi + spread
                st.caption(f"Typical range for {city} in {MONTH_NAMES[month-1]}: **{lo:.0f} – {hi:.0f}** (based on {len(city_month_hist)} past records)")

        st.markdown(
            f"""<div class="infocard" style="border-left: 4px solid {cat_color};">
            <p class="infocard-title" style="color:{cat_color} !important;">Health advice — {pred_category}</p>
            <p>{HEALTH_ADVICE.get(pred_category, "")}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.subheader("What drove this prediction?")
        importances = reg_model.feature_importances_
        imp_df = pd.DataFrame(
            {"feature": row.columns.tolist(), "importance": importances}
        ).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        style_axes(ax, fig)
        ax.barh(imp_df["feature"], imp_df["importance"], color=PRIMARY)
        ax.set_xlabel("Importance", color=TEXT_MAIN)
        ax.set_title("Feature importance (Gradient Boosting model)", color=PRIMARY)
        st.pyplot(fig)

        with st.expander(f"📅 Best months to be outside in {city}"):
            with st.spinner("Scoring all 12 months..."):
                month_rows = []
                for m in range(1, 13):
                    mrow = build_row(city, m, season, is_smog_season, is_crop_burning_season, is_monsoon_season)
                    mpred = reg_model.predict(mrow)[0]
                    month_rows.append({"Month": MONTH_NAMES[m-1], "Predicted AQI": round(mpred, 1)})
                month_df = pd.DataFrame(month_rows).sort_values("Predicted AQI")

            fig2, ax2 = plt.subplots(figsize=(8, 3.5))
            style_axes(ax2, fig2)
            bar_colors = [PRIMARY if mn == month_df.iloc[0]["Month"] else f"{PRIMARY}55" for mn in month_df["Month"]]
            ax2.bar(month_df["Month"], month_df["Predicted AQI"], color=bar_colors)
            ax2.set_ylabel("Predicted AQI", color=TEXT_MAIN)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig2)
            st.caption(f"Cleanest predicted month: **{month_df.iloc[0]['Month']}** ({month_df.iloc[0]['Predicted AQI']:.0f} AQI) — with the season and toggles held fixed and month varying.")
    else:
        st.info("Set your inputs in the sidebar, then click **Predict AQI**.")

# ===================== MAP =====================
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

    if map_df["lat"].notna().all():
        st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), size=8000)
    else:
        st.warning("Some cities have no latitude or longitude in city_lookup.json, so only the table is shown.")

    sorted_map_df = map_df.sort_values("Predicted AQI", ascending=False)
    st.dataframe(sorted_map_df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download this table as CSV", sorted_map_df.to_csv(index=False),
                        file_name="aqi_map_all_cities.csv", mime="text/csv")

# ===================== HISTORICAL TREND =====================
elif active == "trend":
    city_hist = history[history["city"] == trend_city].copy()
    city_hist["date"] = pd.to_datetime(
        city_hist["year"].astype(str) + "-" + city_hist["month"].astype(str) + "-01"
    )
    city_hist = city_hist.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 4))
    style_axes(ax, fig)
    ax.plot(city_hist["date"], city_hist["aqi_us"], color=PRIMARY, linewidth=1.6, label="Recorded AQI")
    ax.fill_between(city_hist["date"], city_hist["aqi_us"], color=PRIMARY, alpha=0.12)

    if show_prediction:
        cur_month = pd.Timestamp.today().month
        default_season = SEASONS[0]
        trend_row = build_row(trend_city, cur_month, default_season,
                               cur_month in (11, 12, 1), False, cur_month in (7, 8, 9))
        trend_pred = reg_model.predict(trend_row)[0]
        last_date = city_hist["date"].max()
        ax.scatter([last_date], [trend_pred], color="#C9840A", s=90, zorder=5,
                    label=f"Model's current-month prediction ({trend_pred:.0f})")
        ax.legend(facecolor="#FFFFFF", edgecolor=BORDER_COLOR, labelcolor=TEXT_MAIN, loc="upper left")

    ax.set_ylabel("AQI", color=TEXT_MAIN)
    ax.set_title(f"AQI trend: {trend_city} (2015–2025)", color=PRIMARY)
    ax.grid(alpha=0.15, color=TEXT_MAIN)
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Average AQI", f"{city_hist['aqi_us'].mean():.1f}")
    c2.metric("Peak AQI", f"{city_hist['aqi_us'].max():.1f}")
    c3.metric("Lowest AQI", f"{city_hist['aqi_us'].min():.1f}")

    st.subheader(f"Worst AQI days on record — {trend_city}")
    worst = city_hist.sort_values("aqi_us", ascending=False).head(5)[["date", "aqi_us"]].copy()
    worst["date"] = worst["date"].dt.strftime("%B %Y")
    worst = worst.rename(columns={"date": "Month", "aqi_us": "AQI"})
    st.dataframe(worst, use_container_width=True, hide_index=True)

# ===================== COMPARE CITIES =====================
elif active == "compare":
    if len(compare_cities) >= 2:
        cmp_rows = []
        for c in compare_cities:
            row = build_row(c, cmp_month, cmp_season, cmp_smog, False, False)
            pred = reg_model.predict(row)[0]
            cat = clf_model.predict(row)[0]
            cmp_rows.append({"City": c, "Predicted AQI": round(pred, 1), "Category": cat})

        cmp_df = pd.DataFrame(cmp_rows)

        fig, ax = plt.subplots(figsize=(8, 4))
        style_axes(ax, fig)
        colors = [CATEGORY_COLORS.get(cat, PRIMARY) for cat in cmp_df["Category"]]
        bars = ax.bar(cmp_df["City"], cmp_df["Predicted AQI"], color=colors)
        ax.bar_label(bars, fmt="%.0f", color=TEXT_MAIN, padding=3)
        ax.set_ylabel("Predicted AQI", color=TEXT_MAIN)
        ax.set_title("City comparison", color=PRIMARY)
        st.pyplot(fig)

        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download this table as CSV", cmp_df.to_csv(index=False),
                            file_name="aqi_city_comparison.csv", mime="text/csv")
    else:
        st.info("Pick at least 2 cities in the sidebar to compare.")

# ===================== SEASONAL HEATMAP =====================
elif active == "heatmap":
    with st.spinner("Scoring every city across all 12 months..."):
        grid = np.zeros((len(CITIES), 12))
        for i, c in enumerate(CITIES):
            for m in range(1, 13):
                row = build_row(c, m, hm_season, hm_smog, hm_crop, hm_monsoon)
                grid[i, m - 1] = reg_model.predict(row)[0]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    im = ax.imshow(grid, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=max(300, grid.max()))
    ax.set_xticks(range(12))
    ax.set_xticklabels([m[:3] for m in MONTH_NAMES], color=TEXT_MAIN)
    ax.set_yticks(range(len(CITIES)))
    ax.set_yticklabels(CITIES, color=TEXT_MAIN)
    for i in range(len(CITIES)):
        for j in range(12):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                     color="#0B1A21", fontsize=8, fontweight="bold")
    ax.set_title("Predicted AQI — every city × every month", color=PRIMARY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=TEXT_MAIN)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_MAIN)
    st.pyplot(fig)

    heat_df = pd.DataFrame(grid, index=CITIES, columns=MONTH_NAMES).round(1)
    st.dataframe(heat_df, use_container_width=True)
    st.download_button("⬇️ Download this grid as CSV", heat_df.to_csv(),
                        file_name="aqi_seasonal_heatmap.csv", mime="text/csv")
    st.caption("Season and toggles above are held fixed across all 12 columns — only the month input changes per cell.")

st.divider()
st.caption(
    "Model: hyperparameter-tuned Gradient Boosting (R² = 0.974 regression, 91.7% classification accuracy) — "
    "Pakistan Air Quality Index dataset, 10 cities, 2015–2025."
)
