import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Pakistan AQI Predictor", page_icon="🌫️", layout="wide")

# ==================================================================
#  THEME TOKENS — one accent colour per section, used everywhere
# ==================================================================
ACCENT = {
    "predict": "#22D3EE",   # cyan
    "map":     "#34D399",   # green
    "trend":   "#A78BFA",   # violet
    "compare": "#FBBF24",   # amber
    "heatmap": "#FB7185",   # rose
}
DARKTEXT = {
    "predict": "#04161B",
    "map":     "#04160F",
    "trend":   "#120A24",
    "compare": "#1C1302",
    "heatmap": "#210810",
}
PAGE_BG = {
    "predict": "linear-gradient(165deg, #0B1A21 0%, #12313C 45%, #164A5C 100%)",
    "map":     "linear-gradient(165deg, #0B1E1A 0%, #123A31 45%, #17544A 100%)",
    "trend":   "linear-gradient(165deg, #150F26 0%, #2A2050 45%, #3A2C6B 100%)",
    "compare": "linear-gradient(165deg, #201705 0%, #3D2D0C 45%, #55401A 100%)",
    "heatmap": "linear-gradient(165deg, #200B12 0%, #3A1522 45%, #56202F 100%)",
}
TITLE_GOLD = "#FFC94A"
INK = "#E8F1F8"
MUTED = "#93AEC2"

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
except Exception:                                   # older Streamlit
    active = st.experimental_get_query_params().get("tab", ["predict"])[0]
if active not in NAV_KEYS:
    active = "predict"

A = ACCENT[active]

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

    /* Streamlit's own tab widget is not used any more */
    .stTabs {{ display: none !important; }}

    /* ---------------- masthead ---------------- */
    .masthead {{ padding: 22px 0 6px 0; }}
    .masthead .titleblock {{ display: inline-block; }}
    .masthead .wordmark {{
        font-size: 3.1rem;
        line-height: 1.05;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: {TITLE_GOLD} !important;
        text-shadow: 0 0 28px rgba(255, 201, 74, 0.35);
        margin: 0;
    }}
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

    /* ---------------- navigation buttons ---------------- */
    .navbar {{
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin: 26px 0 8px 0;
    }}
    .navpill {{
        display: inline-block;
        padding: 11px 26px;
        border-radius: 999px;
        font-size: 1rem;
        font-weight: 800;
        text-decoration: none !important;
        border: 2px solid;
        transition: transform .15s ease, box-shadow .15s ease;
    }}
    .navpill:hover {{ transform: translateY(-2px); text-decoration: none !important; }}
    .navpill:focus-visible {{ outline: 3px solid {TITLE_GOLD}; outline-offset: 3px; }}

    .navpill.predict {{ color: {ACCENT['predict']} !important; background: {ACCENT['predict']}1F; border-color: {ACCENT['predict']}; }}
    .navpill.map     {{ color: {ACCENT['map']} !important;     background: {ACCENT['map']}1F;     border-color: {ACCENT['map']}; }}
    .navpill.trend   {{ color: {ACCENT['trend']} !important;   background: {ACCENT['trend']}26;   border-color: {ACCENT['trend']}; }}
    .navpill.compare {{ color: {ACCENT['compare']} !important; background: {ACCENT['compare']}26; border-color: {ACCENT['compare']}; }}
    .navpill.heatmap {{ color: {ACCENT['heatmap']} !important; background: {ACCENT['heatmap']}26; border-color: {ACCENT['heatmap']}; }}

    .navpill.predict.on {{ color: {DARKTEXT['predict']} !important; background: {ACCENT['predict']}; box-shadow: 0 6px 22px {ACCENT['predict']}70; }}
    .navpill.map.on     {{ color: {DARKTEXT['map']} !important;     background: {ACCENT['map']};     box-shadow: 0 6px 22px {ACCENT['map']}70; }}
    .navpill.trend.on   {{ color: {DARKTEXT['trend']} !important;   background: {ACCENT['trend']};   box-shadow: 0 6px 22px {ACCENT['trend']}70; }}
    .navpill.compare.on {{ color: {DARKTEXT['compare']} !important; background: {ACCENT['compare']}; box-shadow: 0 6px 22px {ACCENT['compare']}70; }}
    .navpill.heatmap.on {{ color: {DARKTEXT['heatmap']} !important; background: {ACCENT['heatmap']}; box-shadow: 0 6px 22px {ACCENT['heatmap']}70; }}

    /* ---------------- section banner ---------------- */
    .panel-head {{
        border-radius: 0 14px 14px 0;
        padding: 16px 20px;
        margin: 26px 0 6px 0;
        border-left: 6px solid {A};
        background: linear-gradient(90deg, {A}26 0%, rgba(255,255,255,0.02) 85%);
    }}
    .panel-head .panel-title {{
        font-size: 1.45rem; font-weight: 800; margin: 0;
        letter-spacing: -0.01em; color: {A} !important;
    }}
    .panel-head .panel-sub {{ color: {MUTED} !important; font-size: .95rem; margin: 4px 0 0 0; }}
    .rule {{ height: 2px; border: 0; margin: 0 0 20px 0; border-radius: 2px; background: {A}; }}

    /* ---------------- metrics ---------------- */
    div[data-testid="stMetric"] {{
        background-color: rgba(255,255,255,0.07);
        border: 1px solid {A}55;
        border-radius: 14px;
        padding: 16px;
    }}
    div[data-testid="stMetricLabel"] p {{ color: {MUTED} !important; font-weight: 600; }}
    div[data-testid="stMetricValue"] {{ color: #FFFFFF !important; }}

    /* ---------------- buttons & inputs ---------------- */
    .stButton>button, .stDownloadButton>button {{
        background: {A};
        color: {DARKTEXT[active]};
        border-radius: 10px;
        font-weight: 800;
        border: none;
        padding: 12px 0px;
        transition: transform .15s ease;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{ transform: translateY(-1px); color: {DARKTEXT[active]}; }}
    .stButton>button:focus-visible, .stDownloadButton>button:focus-visible {{ outline: 3px solid {TITLE_GOLD}; outline-offset: 2px; }}

    div[data-baseweb="select"] > div {{
        background-color: rgba(255,255,255,0.08) !important;
        border: 1px solid {A}55 !important;
        border-radius: 10px;
        color: {INK} !important;
    }}
    div[data-baseweb="select"] div, div[data-baseweb="select"] span {{ color: {INK} !important; }}
    div[data-baseweb="popover"] li {{ color: #0B1A21 !important; }}
    [data-testid="stDataFrame"] {{ border-radius: 10px; }}

    /* ---------------- info / advice / city cards ---------------- */
    .infocard {{
        border-radius: 14px;
        padding: 18px 20px;
        margin: 14px 0;
        border: 1px solid {A}55;
        background: rgba(255,255,255,0.05);
    }}
    .infocard .infocard-title {{
        font-weight: 800; font-size: 1.02rem; margin: 0 0 8px 0; color: {A} !important;
    }}
    .infocard p {{ margin: 2px 0; }}
    .chip-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }}
    .chip {{
        font-size: .82rem; font-weight: 700; padding: 4px 12px;
        border-radius: 999px; background: {A}26; color: {A} !important; border: 1px solid {A}55;
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

# AQI breakpoints used for the gauge bands (US EPA scale)
AQI_BANDS = [
    (0,   50,  "#2ECC71", "Good"),
    (50,  100, "#F1C40F", "Moderate"),
    (100, 150, "#E67E22", "Unhealthy for Sensitive Groups"),
    (150, 200, "#E67E22", "Unhealthy"),
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
    """Half-circle AQI gauge, 0-500, coloured bands + needle."""
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


panel()

# ===================== PREDICT =====================
if active == "predict":
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

    with col2:
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
                st.caption(f"Cleanest predicted month: **{month_df.iloc[0]['Month']}** ({month_df.iloc[0]['Predicted AQI']:.0f} AQI) — with the season and toggles above held fixed and month varying.")
        else:
            st.info("Set your inputs on the left, then click **Predict AQI**.")

# ===================== MAP =====================
elif active == "map":
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
        st.map(map_df.rename(columns={"lat": "latitude", "lon": "longitude"}), size=14000, color=A)
    else:
        st.warning("Some cities have no latitude or longitude in city_lookup.json, so only the table is shown.")

    sorted_map_df = map_df.sort_values("Predicted AQI", ascending=False)
    st.dataframe(sorted_map_df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download this table as CSV", sorted_map_df.to_csv(index=False),
                        file_name="aqi_map_all_cities.csv", mime="text/csv")

# ===================== HISTORICAL TREND =====================
elif active == "trend":
    t1, t2 = st.columns(2)
    with t1:
        trend_city = st.selectbox("Select a city", CITIES, key="t_city")
    with t2:
        show_prediction = st.toggle("Overlay this month's prediction", value=True, key="t_overlay")

    city_hist = history[history["city"] == trend_city].copy()
    city_hist["date"] = pd.to_datetime(
        city_hist["year"].astype(str) + "-" + city_hist["month"].astype(str) + "-01"
    )
    city_hist = city_hist.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 4))
    style_axes(ax, fig)
    ax.plot(city_hist["date"], city_hist["aqi_us"], color=A, linewidth=1.6, label="Recorded AQI")
    ax.fill_between(city_hist["date"], city_hist["aqi_us"], color=A, alpha=0.15)

    if show_prediction:
        cur_month = pd.Timestamp.today().month
        default_season = SEASONS[0]
        trend_row = build_row(trend_city, cur_month, default_season,
                               cur_month in (11, 12, 1), False, cur_month in (7, 8, 9))
        trend_pred = reg_model.predict(trend_row)[0]
        last_date = city_hist["date"].max()
        ax.scatter([last_date], [trend_pred], color=TITLE_GOLD, s=90, zorder=5,
                    label=f"Model's current-month prediction ({trend_pred:.0f})")
        ax.legend(facecolor="none", edgecolor="none", labelcolor=INK, loc="upper left")

    ax.set_ylabel("AQI", color=INK)
    ax.set_title(f"AQI trend: {trend_city} (2015–2025)", color=A)
    ax.grid(alpha=0.15, color=INK)
    st.pyplot(fig)

    c1, c2, c3 = st.columns(3)
    c1.metric("Average AQI", f"{city_hist['aqi_us'].mean():.1f}")
    c2.metric("Peak AQI", f"{city_hist['aqi_us'].max():.1f}")
    c3.metric("Lowest AQI", f"{city_hist['aqi_us'].min():.1f}")

# ===================== COMPARE CITIES =====================
elif active == "compare":
    compare_cities = st.multiselect("Select 2–4 cities", CITIES,
                                    default=CITIES[:2], max_selections=4, key="c_cities")
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
        st.info("Pick at least 2 cities to compare.")

# ===================== SEASONAL HEATMAP =====================
elif active == "heatmap":
    h1, h2 = st.columns(2)
    with h1:
        hm_season = st.selectbox("Season", SEASONS, key="h_season")
    with h2:
        hm_smog = st.toggle("Smog season active?", value=(hm_season == "Winter"), key="h_smog")
        hm_crop = st.toggle("Crop-burning season active?", key="h_crop")
        hm_monsoon = st.toggle("Monsoon season active?", key="h_monsoon")

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
