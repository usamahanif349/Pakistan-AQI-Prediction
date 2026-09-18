import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from realtime_api import get_live_city_aqi

st.set_page_config(page_title="Pakistan AQI Predictor", page_icon="🌫️", layout="wide")

# ==================================================================
#  THEME TOKENS — light background, one distinguishable accent per section
# ==================================================================
ACCENT = {
    "predict": "#0891B2",   # teal
    "map":     "#15803D",   # green
    "trend":   "#7C3AED",   # violet
    "compare": "#B45309",   # amber/ochre
    "heatmap": "#BE123C",   # rose
}
TINT = {           # very light wash of each accent, for cards / sidebar / active pill fill
    "predict": "#E0F6FA",
    "map":     "#E7F6EC",
    "trend":   "#F1EBFC",
    "compare": "#FCF1E1",
    "heatmap": "#FCE9ED",
}
PAGE_BG = {
    "predict": "linear-gradient(165deg, #F3FBFD 0%, #E7F6F9 100%)",
    "map":     "linear-gradient(165deg, #F3FBF5 0%, #E8F6EC 100%)",
    "trend":   "linear-gradient(165deg, #F8F5FE 0%, #EFE8FC 100%)",
    "compare": "linear-gradient(165deg, #FEFAF3 0%, #FBF0DE 100%)",
    "heatmap": "linear-gradient(165deg, #FEF5F7 0%, #FCE9ED 100%)",
}
TITLE_INK = "#0B2A3B"   # dark navy for the wordmark, high contrast on light bg
GOLD = "#C9840A"        # underline colour — warm but not the AI-cliché terracotta
INK = "#1E2E3A"          # body text
MUTED = "#5B7387"        # secondary text

NAV = [
    ("predict", "🔮 Predict",            "Pick a city and month, then read the AQI value and health category."),
    ("map",     "🗺️ Map (All Cities)",   "Every city scored for the same month and season, mapped and ranked."),
    ("trend",   "📈 Historical Trend",   "Recorded AQI for one city, month by month, 2015 to 2025."),
    ("compare", "⚖️ Compare Cities",     "Put two to four cities side by side under identical conditions."),
    ("heatmap", "🔥 Seasonal Heatmap",   "Every city against every month, so the smog season stands out at a glance."),
]
NAV_KEYS = [k for k, _, _ in NAV]

# ---------- which section is open (kept in the URL: ?tab=map) ----------
try:
    active = st.query_params.get("tab", "predict")
except Exception:
    active = st.experimental_get_query_params().get("tab", ["predict"])[0]
if active not in NAV_KEYS:
    active = "predict"

A = ACCENT[active]
T = TINT[active]

# ==================================================================
#  STYLING
# ==================================================================
st.markdown(f"""
<style>
    .stApp {{
        background: {PAGE_BG[active]};
        background-attachment: fixed;
    }}
    h1, h2, h3, h4, p, label, .stMarkdown, .stCaption, li {{ color: {INK} !important; }}

    .stTabs {{ display: none !important; }}

    /* ---------------- masthead ---------------- */
    .masthead {{ padding: 20px 0 4px 0; }}
    .masthead .titleblock {{ display: inline-block; }}
    .masthead .wordmark {{
        font-size: 3rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: {TITLE_INK} !important;
        margin: 0;
    }}
    .masthead .goldrule {{
        display: block;
        height: 5px;
        width: 100%;
        margin: 12px 0 12px 0;
        border-radius: 999px;
        background: {GOLD};
    }}
    .masthead .standfirst {{
        color: {MUTED} !important;
        font-size: 1.02rem;
        max-width: 68ch;
        margin: 0;
    }}

    /* ---------------- navigation buttons ---------------- */
    .navbar {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin: 22px 0 6px 0;
    }}
    .navpill {{
        display: inline-block;
        padding: 10px 24px;
        border-radius: 999px;
        font-size: .98rem;
        font-weight: 800;
        text-decoration: none !important;
        border: 2px solid;
        background: #FFFFFF;
        transition: transform .15s ease, box-shadow .15s ease;
    }}
    .navpill:hover {{ transform: translateY(-2px); text-decoration: none !important; }}
    .navpill:focus-visible {{ outline: 3px solid {GOLD}; outline-offset: 3px; }}

    .navpill.predict {{ color: {ACCENT['predict']} !important; border-color: {ACCENT['predict']}66; }}
    .navpill.map     {{ color: {ACCENT['map']} !important;     border-color: {ACCENT['map']}66; }}
    .navpill.trend   {{ color: {ACCENT['trend']} !important;   border-color: {ACCENT['trend']}66; }}
    .navpill.compare {{ color: {ACCENT['compare']} !important; border-color: {ACCENT['compare']}66; }}
    .navpill.heatmap {{ color: {ACCENT['heatmap']} !important; border-color: {ACCENT['heatmap']}66; }}

    .navpill.predict.on {{ color: #FFFFFF !important; background: {ACCENT['predict']}; box-shadow: 0 6px 18px {ACCENT['predict']}55; }}
    .navpill.map.on     {{ color: #FFFFFF !important; background: {ACCENT['map']};     box-shadow: 0 6px 18px {ACCENT['map']}55; }}
    .navpill.trend.on   {{ color: #FFFFFF !important; background: {ACCENT['trend']};   box-shadow: 0 6px 18px {ACCENT['trend']}55; }}
    .navpill.compare.on {{ color: #FFFFFF !important; background: {ACCENT['compare']}; box-shadow: 0 6px 18px {ACCENT['compare']}55; }}
    .navpill.heatmap.on {{ color: #FFFFFF !important; background: {ACCENT['heatmap']}; box-shadow: 0 6px 18px {ACCENT['heatmap']}55; }}

    /* ---------------- section banner ---------------- */
    .panel-head {{
        border-radius: 0 14px 14px 0;
        padding: 14px 20px;
        margin: 24px 0 6px 0;
        border-left: 6px solid {A};
        background: {T};
    }}
    .panel-head .panel-title {{
        font-size: 1.4rem; font-weight: 800; margin: 0;
        letter-spacing: -0.01em; color: {A} !important;
    }}
    .panel-head .panel-sub {{ color: {MUTED} !important; font-size: .93rem; margin: 4px 0 0 0; }}
    .rule {{ height: 2px; border: 0; margin: 0 0 18px 0; border-radius: 2px; background: {A}; }}

    /* ---------------- sidebar ---------------- */
    section[data-testid="stSidebar"] {{
        background: {T};
        border-right: 1px solid {A}33;
    }}
    section[data-testid="stSidebar"] * {{ color: {INK} !important; }}
    .sb-heading {{
        font-weight: 800; font-size: 1.05rem; color: {A} !important;
        margin: 4px 0 14px 0; padding-bottom: 8px; border-bottom: 2px solid {A}55;
    }}

    /* ---------------- metrics ---------------- */
    div[data-testid="stMetric"] {{
        background-color: #FFFFFF;
        border: 1px solid {A}44;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 2px 10px rgba(11,42,59,0.05);
    }}
    div[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-weight: 600; }}
    div[data-testid="stMetricValue"] {{ color: {TITLE_INK} !important; }}

    /* ---------------- buttons & inputs ---------------- */
    .stButton>button, .stDownloadButton>button {{
        background: {A};
        color: #FFFFFF;
        border-radius: 10px;
        font-weight: 800;
        border: none;
        padding: 12px 0px;
        transition: transform .15s ease;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{ transform: translateY(-1px); color: #FFFFFF; }}
    .stButton>button:focus-visible, .stDownloadButton>button:focus-visible {{ outline: 3px solid {GOLD}; outline-offset: 2px; }}

    div[data-baseweb="select"] > div {{
        background-color: #FFFFFF !important;
        border: 1px solid {A}55 !important;
        border-radius: 10px;
        color: {INK} !important;
    }}
    div[data-baseweb="select"] div, div[data-baseweb="select"] span {{ color: {INK} !important; }}
    div[data-baseweb="popover"] li {{ color: {INK} !important; }}
    [data-testid="stDataFrame"] {{ border-radius: 10px; }}

    /* ---------------- info / advice / city cards ---------------- */
    .infocard {{
        border-radius: 14px;
        padding: 16px 18px;
        margin: 12px 0;
        border: 1px solid {A}44;
        background: #FFFFFF;
        box-shadow: 0 2px 10px rgba(11,42,59,0.05);
    }}
    .infocard .infocard-title {{
        font-weight: 800; font-size: 1rem; margin: 0 0 6px 0; color: {A} !important;
    }}
    .infocard p {{ margin: 2px 0; }}
    .chip-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
    .chip {{
        font-size: .8rem; font-weight: 700; padding: 4px 12px;
        border-radius: 999px; background: {T}; color: {A} !important; border: 1px solid {A}55;
    }}
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

# ---------- Navigation ----------
pills = "".join(
    f'<a class="navpill {key}{" on" if key == active else ""}" href="?tab={key}" target="_self">{label}</a>'
    for key, label, _ in NAV
)
st.markdown(f'<div class="navbar">{pills}</div>', unsafe_allow_html=True)


def panel():
    label = next(l for k, l, _ in NAV if k == active)
    sub = next(s for k, _, s in NAV if k == active)
    st.markdown(
        f'<div class="panel-head"><p class="panel-title">{label}</p>'
        f'<p class="panel-sub">{sub}</p></div><hr class="rule">',
        unsafe_allow_html=True,
    )


def style_axes(ax, fig):
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color(A)
        spine.set_alpha(0.7)


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
    "Good": "#22A55A",
    "Moderate": "#D4A72C",
    "Unhealthy": "#E27D2E",
    "Unhealthy for Sensitive Groups": "#E27D2E",
    "Very Unhealthy": "#C0392B",
    "Hazardous": "#7D3C98"
}

AQI_BANDS = [
    (0,   50,  "#22A55A", "Good"),
    (50,  100, "#D4A72C", "Moderate"),
    (100, 150, "#E27D2E", "Unhealthy for Sensitive Groups"),
    (150, 200, "#E27D2E", "Unhealthy"),
    (200, 300, "#C0392B", "Very Unhealthy"),
    (300, 500, "#7D3C98", "Hazardous"),
]

HEALTH_ADVICE = {
    "Good": "Air quality is satisfactory. Enjoy normal outdoor activity.",
    "Moderate": "Acceptable air quality. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Children, the elderly, and people with asthma or heart conditions should limit prolonged outdoor exertion.",
    "Unhealthy": "Everyone may begin to feel effects. Limit prolonged outdoor exertion, especially for sensitive groups.",
    "Very Unhealthy": "Health alert. Avoid outdoor exertion; sensitive groups should stay indoors and use an air purifier if possible.",
    "Hazardous": "Health emergency. Stay indoors, keep windows closed, and wear an N95 mask if you must go outside.",
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
             color=INK, linewidth=3, solid_capstyle="round")
    ax.add_patch(plt.matplotlib.patches.Circle((0, 0), 0.05, color=INK))

    ax.text(0, -0.28, f"{value:.0f}", ha="center", va="center",
             fontsize=26, fontweight="bold", color=color)
    ax.text(0, -0.5, "AQI", ha="center", va="center", fontsize=11, color=MUTED)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.55, 1.1)
    ax.axis("off")
    return fig


# ==================================================================
#  SIDEBAR — inputs live here, scoped to the active section
# ==================================================================
with st.sidebar:
    if active == "predict":
        st.markdown('<p class="sb-heading">🔮 Predict — inputs</p>', unsafe_allow_html=True)
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
            <div class="infocard">
              <p class="infocard-title">{city}</p>
              <p>Province: {meta.get('province', '—')}</p>
              <p>Population: {meta.get('population_millions', '—')}M</p>
              <div class="chip-row">
                {'<span class="chip">Industrial hub</span>' if meta.get('is_industrial_hub') else ''}
                {'<span class="chip">Coastal</span>' if meta.get('is_coastal') else ''}
                {'<span class="chip">Capital</span>' if meta.get('is_capital') else ''}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif active == "map":
        st.markdown('<p class="sb-heading">🗺️ Map — inputs</p>', unsafe_allow_html=True)
        map_month = month_selector("Month", key="m_month")
        map_season = st.selectbox("Season", SEASONS, key="m_season")
        map_smog = st.toggle("Smog season active?", value=(map_season == "Winter"), key="m_smog")
        map_crop = st.toggle("Crop-burning season active?", key="m_crop")
        map_monsoon = st.toggle("Monsoon season active?", key="m_monsoon")

    elif active == "trend":
        st.markdown('<p class="sb-heading">📈 Trend — inputs</p>', unsafe_allow_html=True)
        trend_city = st.selectbox("Select a city", CITIES, key="t_city")
        show_prediction = st.toggle("Overlay this month's prediction", value=True, key="t_overlay")

    elif active == "compare":
        st.markdown('<p class="sb-heading">⚖️ Compare — inputs</p>', unsafe_allow_html=True)
        compare_cities = st.multiselect("Select 2–4 cities", CITIES,
                                        default=CITIES[:2], max_selections=4, key="c_cities")
        cmp_month = month_selector("Month", key="c_month")
        cmp_season = st.selectbox("Season", SEASONS, key="c_season")
        cmp_smog = st.toggle("Smog season active?", value=(cmp_season == "Winter"), key="c_smog")

    elif active == "heatmap":
        st.markdown('<p class="sb-heading">🔥 Heatmap — inputs</p>', unsafe_allow_html=True)
        hm_season = st.selectbox("Season", SEASONS, key="h_season")
        hm_smog = st.toggle("Smog season active?", value=(hm_season == "Winter"), key="h_smog")
        hm_crop = st.toggle("Crop-burning season active?", key="h_crop")
        hm_monsoon = st.toggle("Monsoon season active?", key="h_monsoon")


panel()

# ===================== PREDICT =====================
if active == "predict":
    # --- Live Data Fetch ---
    live_data = get_live_city_aqi(city)
    if live_data and live_data.get("live_aqi") is not None:
        st.subheader(f"🌐 Live Monitoring Feed: {city}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Live Observed AQI", live_data["live_aqi"])
        col2.metric("PM2.5 (µg/m³)", live_data.get("pm25", "N/A"))
        col3.metric("Last Updated", live_data.get("time", "N/A"))
        st.divider()

    if predict_btn:
        row = build_row(city, month, season, is_smog_season, is_crop_burning_season, is_monsoon_season)
        pred_aqi = reg_model.predict(row)[0]
        pred_category = clf_model.predict(row)[0]
        cat_color = CATEGORY_COLORS.get(pred_category, A)

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
            f"""<div class="infocard" style="border-color:{cat_color}88;">
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
        ax.barh(imp_df["feature"], imp_df["importance"], color=A)
        ax.set_xlabel("Importance", color=INK)
        ax.set_title("Feature importance (Gradient Boosting model)", color=A)
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
            bar_colors = [A if mn == month_df.iloc[0]["Month"] else f"{A}55" for mn in month_df["Month"]]
            ax2.bar(month_df["Month"], month_df["Predicted AQI"], color=bar_colors)
            ax2.set_ylabel("Predicted AQI", color=INK)
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
        st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), size=8000, color=A)
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
    ax.plot(city_hist["date"], city_hist["aqi_us"], color=A, linewidth=1.6, label="Recorded AQI")
    ax.fill_between(city_hist["date"], city_hist["aqi_us"], color=A, alpha=0.12)

    if show_prediction:
        cur_month = pd.Timestamp.today().month
        default_season = SEASONS[0]
        trend_row = build_row(trend_city, cur_month, default_season,
                               cur_month in (11, 12, 1), False, cur_month in (7, 8, 9))
        trend_pred = reg_model.predict(trend_row)[0]
        last_date = city_hist["date"].max()
        ax.scatter([last_date], [trend_pred], color=GOLD, s=90, zorder=5,
                    label=f"Model's current-month prediction ({trend_pred:.0f})")
        ax.legend(facecolor="#FFFFFF", edgecolor=f"{A}44", labelcolor=INK, loc="upper left")

    ax.set_ylabel("AQI", color=INK)
    ax.set_title(f"AQI trend: {trend_city} (2015–2025)", color=A)
    ax.grid(alpha=0.15, color=INK)
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
        colors = [CATEGORY_COLORS.get(cat, A) for cat in cmp_df["Category"]]
        bars = ax.bar(cmp_df["City"], cmp_df["Predicted AQI"], color=colors)
        ax.bar_label(bars, fmt="%.0f", color=INK, padding=3)
        ax.set_ylabel("Predicted AQI", color=INK)
        ax.set_title("City comparison", color=A)
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
    ax.set_xticklabels([m[:3] for m in MONTH_NAMES], color=INK)
    ax.set_yticks(range(len(CITIES)))
    ax.set_yticklabels(CITIES, color=INK)
    for i in range(len(CITIES)):
        for j in range(12):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                     color="#0B1A21", fontsize=8, fontweight="bold")
    ax.set_title("Predicted AQI — every city × every month", color=A)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=INK)
    plt.setp(cbar.ax.get_yticklabels(), color=INK)
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
