import streamlit as st
import joblib
import json
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
from realtime_api import get_live_city_aqi

# Page configuration
st.set_page_config(
    page_title="PakAir Intelligence | Environmental Monitoring", 
    page_icon="pakair_icon.png", 
    layout="wide"
)

# ==================================================================
#  SESSION STATE & USER PREFERENCES
# ==================================================================
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "alert_phone" not in st.session_state:
    st.session_state.alert_phone = ""
if "alerts_enabled" not in st.session_state:
    st.session_state.alerts_enabled = True
if "alert_threshold" not in st.session_state:
    st.session_state.alert_threshold = 150
if "home_city" not in st.session_state:
    st.session_state.home_city = "Lahore"

# ==================================================================
#  DESIGN TOKENS — Enterprise Dark Slate Theme
# ==================================================================
PRIMARY = "#0F172A"       # Deep slate navy
BORDER_COLOR = "#CBD5E1"  # Structured gray border
CARD_BG = "#FFFFFF"       # White card surfaces
TEXT_MAIN = "#0F172A"     # High-contrast primary text
TEXT_MUTED = "#64748B"    # Secondary text

NAV = [
    ("predict", "Predict",          "Real-time feed & predictive engine"),
    ("map",     "Spatial Map",      "Geographical pollution scoring"),
    ("trend",   "Historical Trend", "2015–2025 seasonal analytics"),
    ("compare", "Multi-Compare",    "Multi-city scenario modeling"),
    ("heatmap", "Smog Heatmap",     "12-month intensity matrix"),
    ("about",   "Methodology",      "Model architecture & dataset docs"),
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
        background: radial-gradient(circle at 8% 0%, rgba(37,99,235,0.055), transparent 30%), linear-gradient(180deg, #F1F5F9 0%, #F7F9FC 45%, #EEF3F8 100%);
    }}
    h1, h2, h3, h4, p, label, .stMarkdown, .stCaption, li {{ 
        color: {TEXT_MAIN} !important; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    .stTabs {{ display: none !important; }}

    /* Keep the custom masthead fully below Streamlit's fixed top chrome. */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        box-shadow: none !important;
        z-index: 1 !important;
    }}
    .stAppViewContainer .main .block-container {{
        padding-top: 5.2rem !important;
    }}

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
        grid-template-columns: repeat(6, 1fr);
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
        font-size: 0.88rem;
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

    /* Metric Font Fix */
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

    /* Product-grade UI upgrade */
    .brand-wrap {{ display:flex; align-items:center; gap:14px; }}
    .masthead-logo {{ width:54px; height:54px; object-fit:contain; display:block; flex:none; margin-top:0; }}
    .masthead-logo-wrap {{ display:flex; align-items:center; justify-content:center; width:64px; min-width:64px; height:64px; overflow:visible; padding-top:2px; box-sizing:border-box; }}
    .eyebrow {{ font-size:.70rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:#2563EB !important; margin-bottom:3px; }}
    .status-strip {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:0 0 20px; }}
    .status-pill {{ background:#FFFFFF; border:1px solid #E2E8F0; border-radius:11px; padding:10px 12px; min-height:56px; }}
    .status-pill .label {{ display:block; font-size:.66rem; text-transform:uppercase; letter-spacing:.08em; color:#64748B !important; font-weight:800; }}
    .status-pill .value {{ display:block; margin-top:3px; font-size:.86rem; color:#0F172A !important; font-weight:800; }}
    .online-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; background:#10B981; margin-right:6px; vertical-align:middle; box-shadow:0 0 0 3px rgba(16,185,129,.12); }}
    .page-intro {{ display:flex; justify-content:space-between; align-items:flex-end; gap:20px; margin:4px 0 18px; }}
    .page-intro h2 {{ margin:0 !important; font-size:1.55rem !important; letter-spacing:-.025em; }}
    .page-intro p {{ margin:4px 0 0 !important; color:#64748B !important; font-size:.92rem; }}
    .hero-meta {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin:-8px 0 20px; }}
    .hero-meta-card {{ background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:12px 14px; }}
    .hero-meta-label {{ font-size:.66rem; color:#64748B !important; text-transform:uppercase; letter-spacing:.08em; font-weight:800; }}
    .hero-meta-value {{ margin-top:3px; font-size:.94rem; color:#0F172A !important; font-weight:800; }}
    .section-kicker {{ font-size:.70rem; font-weight:800; color:#2563EB !important; letter-spacing:.10em; text-transform:uppercase; margin:24px 0 5px; }}
    .section-heading {{ font-size:1.28rem; font-weight:850; color:#0F172A !important; margin:0 0 12px; }}
    .insight-card {{ background:linear-gradient(135deg,#FFFFFF 0%,#F8FAFC 100%); border:1px solid #E2E8F0; border-radius:14px; padding:15px 17px; min-height:103px; box-shadow:0 2px 8px rgba(15,23,42,.035); }}
    .insight-card .k {{ color:#64748B !important; font-size:.69rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; }}
    .insight-card .v {{ color:#0F172A !important; font-size:1.38rem; font-weight:900; margin-top:4px; }}
    .insight-card .s {{ color:#64748B !important; font-size:.76rem; margin-top:3px; }}
    .legend-row {{ display:flex; flex-wrap:wrap; gap:7px; margin:4px 0 14px; }}
    .legend-item {{ display:inline-flex; align-items:center; gap:6px; padding:6px 9px; border-radius:999px; background:#FFFFFF; border:1px solid #E2E8F0; color:#334155 !important; font-size:.71rem; font-weight:700; }}
    .legend-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
    .callout {{ border:1px solid #BFDBFE; background:#EFF6FF; border-radius:12px; padding:12px 14px; margin:12px 0; color:#334155 !important; }}
    .callout strong {{ color:#1D4ED8 !important; }}
    .arch-step {{ background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:12px; text-align:center; min-height:86px; }}
    .arch-step .num {{ display:inline-flex; align-items:center; justify-content:center; width:25px; height:25px; border-radius:50%; background:#DBEAFE; color:#1D4ED8 !important; font-weight:900; font-size:.75rem; margin-bottom:6px; }}
    .arch-step .title {{ display:block; color:#0F172A !important; font-weight:850; font-size:.82rem; }}
    .arch-step .desc {{ display:block; color:#64748B !important; font-size:.70rem; margin-top:3px; }}
    .footer-line {{ margin-top:36px; padding:18px 0 8px; border-top:1px solid #E2E8F0; color:#94A3B8 !important; font-size:.70rem; text-align:center; }}
    /* Subtle environmental-tech canvas: more depth without making the UI noisy */
    .block-container {{ padding-top: 1.2rem !important; padding-bottom: 2.2rem !important; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color:#E2E8F0 !important; border-radius:14px !important; }}
    div[data-testid="stDataFrame"] {{ border:1px solid #E2E8F0; border-radius:12px; overflow:hidden; box-shadow:0 2px 8px rgba(15,23,42,.035); }}
    div[data-testid="stDownloadButton"] button {{ border-radius:9px !important; font-weight:750 !important; border:1px solid #CBD5E1 !important; }}
    div.stButton > button {{ border-radius:9px !important; font-weight:800 !important; min-height:42px; }}
    div[data-testid="stExpander"] {{ border:1px solid #E2E8F0 !important; border-radius:12px !important; background:rgba(255,255,255,.68) !important; }}
    .nav-bar-container {{ backdrop-filter: blur(8px); }}
    .infocard, div[data-testid="stMetric"], .insight-card, .hero-meta-card, .status-pill {{ backdrop-filter: blur(4px); }}
    .section-heading {{ letter-spacing:-.02em; }}

    @media (max-width:900px) {{ .status-strip{{grid-template-columns:repeat(2,1fr)}} .hero-meta{{grid-template-columns:1fr}} .nav-bar-container{{grid-template-columns:repeat(2,1fr)}} }}
</style>
""", unsafe_allow_html=True)

# ---------- Top Masthead ----------
# Render the masthead logo as inline HTML so it cannot be clipped by a narrow
# Streamlit column or overlap the app's top chrome.
try:
    with open("pakair_icon.png", "rb") as _logo_file:
        _logo_b64 = base64.b64encode(_logo_file.read()).decode("utf-8")
except Exception:
    _logo_b64 = ""

head_left, head_right = st.columns([5, 1], vertical_alignment="center")
with head_left:
    logo_col, brand_col = st.columns([0.95, 7])
    with logo_col:
        st.markdown(
            f'<div class="masthead-logo-wrap"><img class="masthead-logo" src="data:image/png;base64,{_logo_b64}" alt="PakAir Intelligence logo"></div>',
            unsafe_allow_html=True,
        )
    with brand_col:
        st.markdown("""
        <div class="eyebrow">Environmental Intelligence Platform</div>
        <h1 class="brand-title">PakAir Intelligence</h1>
        <p class="brand-subtitle">Real-Time Environmental Monitoring & Predictive Analytics for Pakistan</p>
        """, unsafe_allow_html=True)
with head_right:
    st.markdown("""
    <div style="text-align:right;padding-top:8px;">
        <div style="font-size:.66rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;color:#64748B;">Platform Status</div>
        <div style="margin-top:5px;font-size:.84rem;font-weight:800;color:#0F172A;"><span class="online-dot"></span>Operational</div>
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

st.markdown(f"""
<div class="status-strip">
  <div class="status-pill"><span class="label">Coverage</span><span class="value">{len(CITIES)} Pakistani cities</span></div>
  <div class="status-pill"><span class="label">Historical window</span><span class="value">{int(history['year'].min())}–{int(history['year'].max())}</span></div>
  <div class="status-pill"><span class="label">Prediction engine</span><span class="value">Gradient Boosting</span></div>
  <div class="status-pill"><span class="label">Live feed</span><span class="value"><span class="online-dot"></span>WAQI integration</span></div>
</div>
""", unsafe_allow_html=True)
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
        return "#8B5CF6", "Very Unhealthy", "!"
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

def page_intro(title, subtitle, kicker="PakAir Intelligence"):
    st.markdown(f"""<div class="page-intro"><div><div class="eyebrow">{kicker}</div><h2>{title}</h2><p>{subtitle}</p></div></div>""", unsafe_allow_html=True)

def insight_card(label, value, subtitle=""):
    st.markdown(f"""<div class="insight-card"><div class="k">{label}</div><div class="v">{value}</div><div class="s">{subtitle}</div></div>""", unsafe_allow_html=True)

def add_footer():
    st.markdown("""<div class="footer-line">PakAir Intelligence · Environmental Monitoring & Predictive Analytics · Prototype Platform · Historical analytics + live WAQI integration</div>""", unsafe_allow_html=True)

# ==================================================================
#  SIDEBAR — CONTROLS & USER SETTINGS
# ==================================================================
with st.sidebar:
    st.image("pakair_icon.png", width=42)
    st.markdown("<div style=\"font-size:1.05rem;font-weight:850;color:#F8FAFC;margin:-2px 0 3px;\">PakAir Intelligence</div><div style=\"font-size:.72rem;color:#94A3B8;margin-bottom:18px;\">Pakistan Air Quality Intelligence</div>", unsafe_allow_html=True)
    if active == "predict":
        st.markdown('<p class="sb-title">Predict Inputs</p>', unsafe_allow_html=True)
        
        default_city_idx = CITIES.index(st.session_state.home_city) if st.session_state.home_city in CITIES else 0
        city = st.selectbox("City", CITIES, index=default_city_idx, key="p_city")
        month = month_selector("Month", key="p_month")
        season = st.selectbox("Season", SEASONS, key="p_season")
        is_smog_season = st.toggle("Smog season active?", value=(season == "Winter"), key="p_smog")
        is_crop_burning_season = st.toggle("Crop-burning season active?", key="p_crop")
        is_monsoon_season = st.toggle("Monsoon season active?", key="p_monsoon")
        predict_btn = st.button("Predict AQI", type="primary", use_container_width=True)

        meta = city_lookup[city]
        st.markdown(
            f"""
            <div style="background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-top: 16px;">
              <p style="color: #FFFFFF !important; font-weight: 800; font-size: 1.1rem; margin: 0 0 6px 0;">{city}</p>
              <p style="color: #CBD5E1 !important; font-size: 0.9rem; margin: 2px 0;">Province: {meta.get('province', '—')}</p>
              <p style="color: #CBD5E1 !important; font-size: 0.9rem; margin: 2px 0;">Population: {meta.get('population_millions', '—')}M</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="sb-title" style="margin-top: 24px;">Alert Preferences</p>', unsafe_allow_html=True)
        st.session_state.alerts_enabled = st.toggle("Enable Unhealthy Alerts", value=st.session_state.alerts_enabled)
        st.session_state.user_email = st.text_input(
            "Alert Email", 
            value=st.session_state.user_email,
            placeholder="e.g. name@example.com", autocomplete="off"
        )
        st.session_state.alert_phone = st.text_input(
            "Alert SMS Phone", 
            value=st.session_state.alert_phone,
            placeholder="e.g. +92 300 1234567", autocomplete="off"
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

    elif active == "about":
        st.markdown('<p class="sb-title">System Specs</p>', unsafe_allow_html=True)
        st.caption("Version: 2.1.0-Enterprise")
        st.caption("Engine: Gradient Boosting Regressor")
        st.caption("API: WAQI Feed Standard")

# ===================== PREDICT =====================
if active == "predict":
    page_intro("Air Quality Command Center", "Monitor current conditions, inspect health status, and run an AI-powered AQI prediction.")
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

        st.markdown(f"""
        <div class="hero-meta">
          <div class="hero-meta-card"><div class="hero-meta-label">PM2.5</div><div class="hero-meta-value">{live_data.get('pm25', 'N/A')} µg/m³</div></div>
          <div class="hero-meta-card"><div class="hero-meta-label">Monitoring Station</div><div class="hero-meta-value">{live_data.get('station', city)}</div></div>
          <div class="hero-meta-card"><div class="hero-meta-label">Health Guidance</div><div class="hero-meta-value">{HEALTH_ADVICE.get(cat_label, '')}</div></div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.alerts_enabled and aqi_val > st.session_state.alert_threshold:
            recipient_info = []
            if st.session_state.user_email:
                recipient_info.append(f"`{st.session_state.user_email}`")
            if st.session_state.alert_phone:
                recipient_info.append(f"`{st.session_state.alert_phone}`")
            
            target_str = " and ".join(recipient_info) if recipient_info else "registered alert channels"

            st.warning(
                f"🚨 **Air Quality Alert Triggered**: Live AQI ({aqi_val}) in {city} exceeds the health threshold ({st.session_state.alert_threshold}). "
                f"Simulated notification dispatched to {target_str}."
            )

    if predict_btn:
        row = build_row(city, month, season, is_smog_season, is_crop_burning_season, is_monsoon_season)
        pred_aqi = reg_model.predict(row)[0]
        pred_category = clf_model.predict(row)[0]
        cat_color = CATEGORY_COLORS.get(pred_category, PRIMARY)

        st.markdown('<div class="section-kicker">AI inference</div><div class="section-heading">AQI Prediction Result</div>', unsafe_allow_html=True)
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
            <p class="infocard-title" style="color:{cat_color} !important; font-weight:800;">Health advice — {pred_category}</p>
            <p>{HEALTH_ADVICE.get(pred_category, "")}</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(f"""<div class="callout"><strong>Prediction context</strong><br>This estimate uses the selected city, month, season, and environmental-season indicators. It is a model output and should be interpreted alongside the live monitoring feed when available.</div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-kicker">Model transparency</div><div class="section-heading">What drove this prediction?</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-kicker">AQI reference</div><div class="section-heading">Health Category Guide</div>', unsafe_allow_html=True)
    st.markdown("""<div class="legend-row">
      <span class="legend-item"><span class="legend-dot" style="background:#10B981"></span>0–50 Good</span>
      <span class="legend-item"><span class="legend-dot" style="background:#F59E0B"></span>51–100 Moderate</span>
      <span class="legend-item"><span class="legend-dot" style="background:#F97316"></span>101–150 USG</span>
      <span class="legend-item"><span class="legend-dot" style="background:#EF4444"></span>151–200 Unhealthy</span>
      <span class="legend-item"><span class="legend-dot" style="background:#8B5CF6"></span>201–300 Very Unhealthy</span>
      <span class="legend-item"><span class="legend-dot" style="background:#6B21A8"></span>301+ Hazardous</span>
    </div>""", unsafe_allow_html=True)

# ===================== MAP (INTERACTIVE GEOSPATIAL DASHBOARD) =====================
elif active == "map":
    page_intro("Pakistan Air Quality Map", "Explore predicted AQI conditions across the monitored city network.")
    map_rows = []
    for c in CITIES:
        row = build_row(c, map_month, map_season, map_smog, map_crop, map_monsoon)
        pred = reg_model.predict(row)[0]
        cat = clf_model.predict(row)[0]
        meta = city_lookup[c]
        
        if pred <= 50:
            color_rgb = [16, 185, 129, 200]
        elif pred <= 100:
            color_rgb = [245, 158, 11, 200]
        elif pred <= 150:
            color_rgb = [249, 115, 22, 200]
        elif pred <= 200:
            color_rgb = [239, 68, 68, 200]
        elif pred <= 300:
            color_rgb = [139, 92, 246, 200]
        else:
            color_rgb = [107, 33, 168, 200]

        map_rows.append({
            "city": c,
            "province": meta.get("province", "N/A"),
            "latitude": meta.get("latitude"),
            "longitude": meta.get("longitude"),
            "Predicted_AQI": round(pred, 1),
            "Category": cat,
            "color_rgb": color_rgb,
            "elevation": pred * 150
        })

    map_df = pd.DataFrame(map_rows)

    c_m1, c_m2 = st.columns([1, 3])
    with c_m1:
        map_mode = st.radio("Map Visualization Mode", ["2D Bubble Pins", "3D Extrusion Columns", "Regional Heatmap"], key="map_mode_select")

    view_state = pdk.ViewState(
        latitude=30.3753,
        longitude=69.3451,
        zoom=4.8,
        pitch=45 if map_mode == "3D Extrusion Columns" else 0
    )

    layers = []
    if map_mode == "2D Bubble Pins":
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                map_df,
                get_position=["longitude", "latitude"],
                get_fill_color="color_rgb",
                get_radius=25000,
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                radius_scale=1,
                radius_min_pixels=10,
                radius_max_pixels=30,
                get_line_color=[255, 255, 255],
                line_width_min_pixels=1.5,
            )
        )
    elif map_mode == "3D Extrusion Columns":
        layers.append(
            pdk.Layer(
                "ColumnLayer",
                map_df,
                get_position=["longitude", "latitude"],
                get_elevation="elevation",
                elevation_scale=1,
                radius=18000,
                get_fill_color="color_rgb",
                pickable=True,
                auto_highlight=True,
            )
        )
    elif map_mode == "Regional Heatmap":
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                map_df,
                get_position=["longitude", "latitude"],
                get_weight="Predicted_AQI",
                radius_pixels=60,
            )
        )

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            "html": "<b>{city}</b> ({province})<br/>"
                    "Predicted AQI: <b>{Predicted_AQI}</b><br/>"
                    "Status: <b>{Category}</b>",
            "style": {"backgroundColor": "#0F172A", "color": "#FFFFFF", "fontSize": "12px", "borderRadius": "8px"}
        }
    )

    st.markdown('<div class="section-kicker">Network overview</div><div class="section-heading">Predicted AQI Snapshot</div>', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        insight_card("Network average", f"{map_df['Predicted_AQI'].mean():.1f}", "Predicted AQI across monitored cities")
    with mc2:
        highest = map_df.loc[map_df['Predicted_AQI'].idxmax()]
        insight_card("Highest predicted", f"{highest['Predicted_AQI']:.1f}", f"{highest['city']} · {highest['Category']}")
    with mc3:
        lowest = map_df.loc[map_df['Predicted_AQI'].idxmin()]
        insight_card("Lowest predicted", f"{lowest['Predicted_AQI']:.1f}", f"{lowest['city']} · {lowest['Category']}")

    st.markdown("""<div class="legend-row">
      <span class="legend-item"><span class="legend-dot" style="background:#10B981"></span>Good</span>
      <span class="legend-item"><span class="legend-dot" style="background:#F59E0B"></span>Moderate</span>
      <span class="legend-item"><span class="legend-dot" style="background:#F97316"></span>USG</span>
      <span class="legend-item"><span class="legend-dot" style="background:#EF4444"></span>Unhealthy</span>
      <span class="legend-item"><span class="legend-dot" style="background:#8B5CF6"></span>Very Unhealthy</span>
      <span class="legend-item"><span class="legend-dot" style="background:#6B21A8"></span>Hazardous</span>
    </div>""", unsafe_allow_html=True)

    st.pydeck_chart(r)

    st.subheader("Regional City Rankings")
    sorted_map_df = map_df[["city", "province", "Predicted_AQI", "Category"]].sort_values("Predicted_AQI", ascending=False)
    st.dataframe(sorted_map_df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download map data as CSV", sorted_map_df.to_csv(index=False), file_name="pakistan_aqi_spatial_rankings.csv", mime="text/csv")

# ===================== HISTORICAL TREND =====================
elif active == "trend":
    page_intro("Historical Air Quality", "Explore long-term AQI behavior and identify recurring pollution patterns by city.")
    city_hist = history[history["city"] == trend_city].copy()
    city_hist["date"] = pd.to_datetime(
        city_hist["year"].astype(str) + "-" + city_hist["month"].astype(str) + "-01"
    )
    city_hist = city_hist.sort_values("date")

    recent = city_hist.tail(12)
    recent_avg = recent["aqi_us"].mean() if len(recent) else np.nan
    st.markdown('<div class="section-kicker">Historical profile</div><div class="section-heading">Long-Term AQI Snapshot</div>', unsafe_allow_html=True)
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        insight_card("Average AQI", f"{city_hist['aqi_us'].mean():.1f}", "Across available historical records")
    with tc2:
        insight_card("Recent-period average", f"{recent_avg:.1f}", "Last 12 recorded observations")
    with tc3:
        insight_card("Peak AQI", f"{city_hist['aqi_us'].max():.1f}", "Highest recorded value in the selected city")

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
    page_intro("Multi-City Comparison", "Compare model-predicted AQI conditions under the same selected environmental scenario.")
    if len(compare_cities) >= 2:
        cmp_rows = []
        for c in compare_cities:
            row = build_row(c, cmp_month, cmp_season, cmp_smog, False, False)
            pred = reg_model.predict(row)[0]
            cat = clf_model.predict(row)[0]
            cmp_rows.append({"City": c, "Predicted AQI": round(pred, 1), "Category": cat})

        cmp_df = pd.DataFrame(cmp_rows)

        st.markdown('<div class="section-kicker">Scenario output</div><div class="section-heading">Comparison Snapshot</div>', unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            insight_card("Cities selected", str(len(compare_cities)), "Included in this scenario")
        with cc2:
            insight_card("Scenario average", f"{cmp_df['Predicted AQI'].mean():.1f}", f"{cmp_season} · {MONTH_NAMES[cmp_month-1]}")
        with cc3:
            spread = cmp_df['Predicted AQI'].max() - cmp_df['Predicted AQI'].min()
            insight_card("City-to-city spread", f"{spread:.1f}", "Difference between highest and lowest prediction")

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
    page_intro("Seasonal Pollution Heatmap", "Scan predicted AQI intensity across cities and months at a glance.")
    with st.spinner("Scoring every city across all 12 months..."):
        grid = np.zeros((len(CITIES), 12))
        for i, c in enumerate(CITIES):
            for m in range(1, 13):
                row = build_row(c, m, hm_season, hm_smog, hm_crop, hm_monsoon)
                grid[i, m - 1] = reg_model.predict(row)[0]

    st.markdown('<div class="section-kicker">Annual scenario scan</div><div class="section-heading">12-Month × City Intelligence Matrix</div>', unsafe_allow_html=True)
    hc1, hc2, hc3 = st.columns(3)
    with hc1:
        insight_card("Network mean", f"{grid.mean():.1f}", "Mean predicted AQI across all cells")
    with hc2:
        idx = np.unravel_index(np.argmax(grid), grid.shape)
        insight_card("Highest cell", f"{grid[idx]:.1f}", f"{CITIES[idx[0]]} · {MONTH_NAMES[idx[1]]}")
    with hc3:
        idx_min = np.unravel_index(np.argmin(grid), grid.shape)
        insight_card("Lowest cell", f"{grid[idx_min]:.1f}", f"{CITIES[idx_min[0]]} · {MONTH_NAMES[idx_min[1]]}")

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

# ===================== ABOUT & METHODOLOGY =====================
elif active == "about":
    page_intro("Platform Methodology", "Understand the data coverage, feature engineering, predictive engine, and live-data architecture.")
    st.subheader("📚 Platform Architecture & Methodology")

    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        insight_card("Coverage", f"{len(CITIES)} cities", "Pakistan monitoring network")
    with ac2:
        insight_card("Historical data", f"{int(history['year'].min())}–{int(history['year'].max())}", "Model training window")
    with ac3:
        insight_card("Regression", "R² 0.974", "Reported model validation")
    with ac4:
        insight_card("Classification", "91.7%", "Reported accuracy")

    st.markdown('<div class="section-kicker">System architecture</div><div class="section-heading">From environmental data to actionable intelligence</div>', unsafe_allow_html=True)
    a1, a2, a3, a4, a5 = st.columns(5)
    for col, num, title, desc in [
        (a1, "1", "Data", "Historical + live feeds"),
        (a2, "2", "Features", "Temporal + regional signals"),
        (a3, "3", "Model", "Gradient Boosting"),
        (a4, "4", "Prediction", "AQI + health band"),
        (a5, "5", "Insights", "Maps, trends, alerts"),
    ]:
        with col:
            st.markdown(f"""<div class="arch-step"><span class="num">{num}</span><span class="title">{title}</span><span class="desc">{desc}</span></div>""", unsafe_allow_html=True)

    st.markdown("""<div class="callout"><strong>Prototype-to-platform direction</strong><br>PakAir Intelligence combines monitoring, prediction, spatial analysis, historical trends, and alerts in one interface. The current application is a software prototype designed around an expandable environmental monitoring architecture.</div>""", unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. Dataset & Coverage
    The predictive engine is trained on **historical air quality records spanning 2015–2025** across 10 major urban centers in Pakistan:
    * **Punjab:** Lahore, Faisalabad, Rawalpindi, Multan, Gujranwala
    * **Sindh:** Karachi, Hyderabad
    * **KPK & Federal:** Islamabad, Peshawar
    * **Balochistan:** Quetta

    ### 2. Feature Engineering Pipeline
    Each sample includes both temporal indicators and regional environmental features:
    * **Temporal Features:** Month encoding, Season classification, Crop Burning indicator flags (October–November), Smog Season flags (Winter inversion).
    * **Geographic/Demographic Features:** Province encoding, Industrial Hub classification, Coastal indicator, Capital status, Population density (millions).

    ### 3. Model Performance & Validation
    * **Regression Engine:** Hyperparameter-tuned **Gradient Boosting Regressor** ($R^2 = 0.974$, $\text{MAE} = 8.2\text{ AQI}$).
    * **Classification Engine:** **Gradient Boosting Classifier** predicting health risk bands ($91.7\%$ Accuracy).

    ### 4. Live API & EPA Breakpoint Standard
    Real-time feeds are ingested via the **World Air Quality Index (WAQI) API**. Concentrations ($\mu g/m^3$) of $PM_{2.5}$ are dynamically mapped to official **US EPA AQI breakpoints** to ensure standardization across global platforms like IQAir.
    """)

st.divider()
st.caption(
    "Model: hyperparameter-tuned Gradient Boosting (R² = 0.974 regression, 91.7% classification accuracy) — "
    "Pakistan Air Quality Index dataset, 10 cities, 2015–2025."
)

add_footer()

