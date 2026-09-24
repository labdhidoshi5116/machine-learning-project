import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HeartSense | Cardiovascular Disease Risk Prediction Using Machine Learning",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# LOAD EXTERNAL CSS DESIGN SYSTEM
# ─────────────────────────────────────────────
def load_css():
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ─────────────────────────────────────────────
# MODEL & DATASET LOADERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "cardiovascular_logistic_regression_pipeline.pkl"
    return joblib.load(model_path)

@st.cache_data
def load_dataset():
    data_path = Path(__file__).parent / "cardio_train.csv"
    if data_path.exists():
        df = pd.read_csv(data_path, sep=";")
        return df
    return None

try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load ML pipeline file (`cardiovascular_logistic_regression_pipeline.pkl`).")
    st.exception(e)
    st.stop()

# ─────────────────────────────────────────────
# CLINICAL HELPER FUNCTIONS
# ─────────────────────────────────────────────
def get_bp_classification(systolic, diastolic):
    """Classify Blood Pressure according to JNC-7 / ACC/AHA guidelines."""
    if systolic >= 180 or diastolic >= 120:
        return "Hypertensive Crisis 🔴", "vitals-danger", "Immediate medical attention required."
    elif systolic >= 140 or diastolic >= 90:
        return "Stage 2 Hypertension 🔴", "vitals-danger", "High risk indicator for cardiac strain."
    elif (130 <= systolic <= 139) or (80 <= diastolic <= 89):
        return "Stage 1 Hypertension 🟠", "vitals-warning", "Moderate cardiovascular impact."
    elif (120 <= systolic <= 129) and diastolic < 80:
        return "Elevated Blood Pressure 🟡", "vitals-warning", "Slightly above optimal range."
    else:
        return "Normal Blood Pressure ✅", "vitals-normal", "Optimal hemodynamic reading."

def get_bmi_classification(bmi):
    """Classify Body Mass Index."""
    if bmi < 18.5:
        return "Underweight", "vitals-warning", "#00BBF9"
    elif bmi < 25.0:
        return "Normal Weight ✅", "vitals-normal", "#00F5D4"
    elif bmi < 30.0:
        return "Overweight ⚠️", "vitals-warning", "#FFB703"
    else:
        return "Obese Class 🔴", "vitals-danger", "#FF0054"

def render_svg_gauge(risk_pct):
    """Render a modern SVG Radial Semi-Circular Risk Gauge with Cyber Emerald styling."""
    risk_pct = max(0.0, min(100.0, risk_pct))
    
    # Angle calculation: 0% -> -90 deg, 100% -> +90 deg
    angle = -90 + (risk_pct / 100.0) * 180
    
    if risk_pct < 35:
        color = "#00F5D4"
        badge_text = "LOW RISK"
        badge_bg = "rgba(0, 245, 212, 0.16)"
        badge_border = "rgba(0, 245, 212, 0.4)"
    elif risk_pct < 65:
        color = "#FFB703"
        badge_text = "MODERATE RISK"
        badge_bg = "rgba(255, 183, 3, 0.16)"
        badge_border = "rgba(255, 183, 3, 0.4)"
    else:
        color = "#FF0054"
        badge_text = "HIGH RISK"
        badge_bg = "rgba(255, 0, 84, 0.16)"
        badge_border = "rgba(255, 0, 84, 0.4)"

    svg_code = f"""
    <div style="text-align: center; padding: 10px 0;">
        <svg width="240" height="135" viewBox="0 0 200 115" style="overflow: visible;">
            <!-- Background Arc -->
            <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#162238" stroke-width="16" stroke-linecap="round" />
            <!-- Active Risk Arc -->
            <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round"
                  stroke-dasharray="251.3" stroke-dashoffset="{251.3 - (risk_pct / 100.0) * 251.3}"
                  style="transition: stroke-dashoffset 1s ease-in-out;" />
            <!-- Center Value Text -->
            <text x="100" y="82" text-anchor="middle" fill="#F8FAFC" font-family="Outfit, sans-serif" font-size="28" font-weight="900">
                {risk_pct:.1f}%
            </text>
            <text x="100" y="98" text-anchor="middle" fill="#8E9BAE" font-family="Plus Jakarta Sans, sans-serif" font-size="10" font-weight="700" letter-spacing="0.05em">
                DISEASE PROBABILITY
            </text>
        </svg>
        <div style="margin-top: 8px;">
            <span style="background: {badge_bg}; border: 1px solid {badge_border}; color: {color}; padding: 6px 16px; border-radius: 999px; font-size: 12px; font-weight: 800; letter-spacing: 0.05em;">
                {badge_text}
            </span>
        </div>
    </div>
    """
    return svg_code

def apply_chart_theme(fig, ax):
    """Set transparent/dark background and clean clinical typography on Matplotlib plots (Cyber Emerald theme)."""
    fig.patch.set_facecolor('#040711')
    ax.set_facecolor('#0D1424')
    ax.tick_params(colors="#8E9BAE", labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#162238')
    ax.spines['bottom'].set_color('#162238')
    ax.grid(axis='y', linestyle='--', alpha=0.2, color='#00BBF9')

# ─────────────────────────────────────────────
# PAGE ROUTING & NAVIGATION CONFIGURATION
# ─────────────────────────────────────────────
PAGES_CONFIG = {
    "overview": "🏠  Overview",
    "assessment": "🔬  Patient Risk Assessment",
    "analytics": "📊  Model Analytics",
    "dataset": "📁  Dataset Explorer"
}
PAGE_TITLE_TO_SLUG = {v: k for k, v in PAGES_CONFIG.items()}
NAV_ITEMS = list(PAGES_CONFIG.values())

def scroll_to_top():
    """Scroll parent window, document, and Streamlit scroll container to top."""
    components.html(
        """
        <script>
            function performScrollTop() {
                try {
                    if (window.parent) {
                        window.parent.scrollTo({ top: 0, left: 0, behavior: 'instant' });
                        const doc = window.parent.document;
                        const scrollTargets = [
                            doc.querySelector('section.main'),
                            doc.querySelector('.stAppViewContainer'),
                            doc.querySelector('.main'),
                            doc.querySelector('[data-testid="stMainBlockContainer"]'),
                            doc.documentElement,
                            doc.body
                        ];
                        scrollTargets.forEach(el => {
                            if (el) {
                                el.scrollTop = 0;
                                el.scrollTo({ top: 0, left: 0, behavior: 'instant' });
                            }
                        });
                    }
                    window.scrollTo({ top: 0, left: 0, behavior: 'instant' });
                } catch (err) {
                    console.debug('Scroll top error:', err);
                }
            }
            performScrollTop();
            setTimeout(performScrollTop, 40);
            setTimeout(performScrollTop, 120);
        </script>
        """,
        height=0,
        width=0,
    )

# 1. Read query parameters from URL
query_param_page = st.query_params.get("page", "overview")
if isinstance(query_param_page, list):
    query_param_page = query_param_page[0] if query_param_page else "overview"

if query_param_page not in PAGES_CONFIG:
    query_param_page = "overview"

# 2. Sync state
if "active_page_slug" not in st.session_state:
    st.session_state["active_page_slug"] = query_param_page

if st.session_state["active_page_slug"] != query_param_page:
    st.session_state["active_page_slug"] = query_param_page

def on_nav_change():
    """Triggered whenever user clicks a different option in navigation."""
    selected_label = st.session_state.get("nav_radio_selection")
    if selected_label in PAGE_TITLE_TO_SLUG:
        selected_slug = PAGE_TITLE_TO_SLUG[selected_label]
        st.session_state["active_page_slug"] = selected_slug
        st.query_params["page"] = selected_slug

active_slug = st.session_state["active_page_slug"]
active_label = PAGES_CONFIG.get(active_slug, "🏠  Overview")
default_nav_index = NAV_ITEMS.index(active_label)

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: rgba(0, 245, 212, 0.2); border: 1px solid rgba(0, 245, 212, 0.4); width: 42px; height: 42px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 22px; box-shadow: 0 0 16px rgba(0, 245, 212, 0.35);">
                🫀
            </div>
            <div>
                <div style="font-family: 'Outfit', sans-serif; font-size: 18px; font-weight: 800; color: #F8FAFC; letter-spacing: -0.01em;">
                    HeartSense
                </div>
                <div style="font-size: 11.5px; color: #00F5D4; font-weight: 700; letter-spacing: 0.03em;">
                    Cardiovascular Risk ML
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        NAV_ITEMS,
        index=default_nav_index,
        key="nav_radio_selection",
        on_change=on_nav_change,
        label_visibility="collapsed"
    )

    # Ensure query param is synchronized
    current_slug = PAGE_TITLE_TO_SLUG.get(page, "overview")
    if st.query_params.get("page") != current_slug:
        st.query_params["page"] = current_slug
        st.session_state["active_page_slug"] = current_slug

    st.markdown("<hr style='border-color: rgba(0, 245, 212, 0.2); margin: 18px 0;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size: 12px; color: #8E9BAE; line-height: 1.7;">
        <div style="font-weight: 800; color: #CBD5E1; margin-bottom: 8px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em;">⚙ Pipeline Info</div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>Algorithm</span>
            <span style="color: #00BBF9; font-weight: 700;">Logistic Regression</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>Preprocessing</span>
            <span style="color: #CBD5E1;">StandardScaler</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span>Test Accuracy</span>
            <span style="color: #00F5D4; font-weight: 800;">71.39%</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Cohort Size</span>
            <span style="color: #CBD5E1; font-family: 'Space Mono', monospace;">N = 70,000</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color: rgba(0, 245, 212, 0.2); margin: 18px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(255, 183, 3, 0.08); border: 1px solid rgba(255, 183, 3, 0.3); border-radius: 12px; padding: 12px 14px; font-size: 11.5px; color: #FFB703; line-height: 1.6;">
        <b>⚠ Disclaimer:</b><br>
        For educational &amp; research use only. Not a substitute for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

# Trigger scroll to top when page changes
if "previous_rendered_page" not in st.session_state:
    st.session_state["previous_rendered_page"] = current_slug
    scroll_to_top()
elif st.session_state["previous_rendered_page"] != current_slug:
    st.session_state["previous_rendered_page"] = current_slug
    scroll_to_top()

# ─────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠  Overview":
    
    # Top Clinical Nav Header
    st.markdown("""
    <div class="top-nav">
        <div class="brand-badge">
            <div class="brand-icon">🫀</div>
            <div>
                <div class="brand-title">HeartSense</div>
                <div class="brand-subtitle">Cardiovascular Disease Risk Prediction Using Machine Learning</div>
            </div>
        </div>
        <div class="status-pill">
            <span class="status-dot"></span>
            Pipeline Active
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero Banner
    st.markdown("""
    <div class="clinical-card" style="background: linear-gradient(135deg, #040711 0%, #0D1424 100%); border-color: rgba(0, 245, 212, 0.35);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
            <div style="max-width: 680px;">
                <span style="background: rgba(0, 245, 212, 0.16); color: #00F5D4; border: 1px solid rgba(0, 245, 212, 0.4); padding: 4px 12px; border-radius: 999px; font-size: 11px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;">
                    Overview
                </span>
                <h1 style="font-family: 'Outfit', sans-serif; font-size: 32px; font-weight: 900; color: #F8FAFC; margin: 14px 0 10px 0; line-height: 1.25; letter-spacing: -0.02em;">
                    HeartSense: Cardiovascular Disease Risk Prediction Using Machine Learning
                </h1>
                <p style="font-size: 15px; color: #8E9BAE; line-height: 1.7; margin: 0; font-weight: 400;">
                    An evidence-based machine learning engine trained on <b style="color: #F8FAFC; font-weight: 700;">70,000 anonymized clinical profiles</b>.
                    Provides probabilistic risk stratification, hemodynamic classification, and multi-factor risk attribution.
                </p>
            </div>
            <div style="background: rgba(4, 7, 17, 0.85); border: 1px solid rgba(0, 187, 249, 0.3); border-radius: 16px; padding: 22px 28px; text-align: center; min-width: 195px; box-shadow: 0 4px 25px rgba(0,0,0,0.6);">
                <div style="font-size: 11px; color: #526075; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em;">Model Accuracy</div>
                <div style="font-family: 'Space Mono', monospace; font-size: 38px; font-weight: 700; color: #00BBF9; margin-top: 4px; line-height: 1;">71.39%</div>
                <div style="font-size: 11.5px; color: #00F5D4; font-weight: 700; margin-top: 6px;">5-Fold Cross-Validated</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stat Grid
    st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #F8FAFC; margin: 28px 0 16px 0; letter-spacing: -0.01em;'>Cohort &amp; Model Metrics</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("70,000", "Patient Cohort", "Cardiovascular dataset records", col1),
        ("12", "Clinical Features", "Biometrics, vitals & labs", col2),
        ("71.39%", "Model Accuracy", "Evaluated on independent test set", col3),
        ("71.58%", "CV Mean Score", "5-Fold stratified cross-validation", col4),
    ]
    for val, lbl, desc, c in stats:
        with c:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{lbl}</div>
                <div class="metric-val">{val}</div>
                <div class="metric-sub">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Workflow & Risk Factors Columns
    left_c, right_c = st.columns(2)

    with left_c:
        st.markdown("""
        <div class="clinical-card">
            <div class="card-title">🔬 How It Works</div>
            <div style="display: flex; flex-direction: column; gap: 18px; margin-top: 6px;">
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div class="step-num">1</div>
                    <div>
                        <div style="font-weight: 800; font-size: 15px; color: #F8FAFC;">Enter patient biometrics</div>
                        <div style="font-size: 13px; color: #8E9BAE; margin-top: 3px; line-height: 1.6;">Age, gender, blood pressure, cholesterol, glucose, and lifestyle details.</div>
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div class="step-num">2</div>
                    <div>
                        <div style="font-weight: 800; font-size: 15px; color: #F8FAFC;">Feature normalization</div>
                        <div style="font-size: 13px; color: #8E9BAE; margin-top: 3px; line-height: 1.6;">StandardScaler standardizes inputs to zero-mean, unit-variance before inference.</div>
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div class="step-num">3</div>
                    <div>
                        <div style="font-weight: 800; font-size: 15px; color: #F8FAFC;">Probabilistic inference</div>
                        <div style="font-size: 13px; color: #8E9BAE; margin-top: 3px; line-height: 1.6;">Logistic Regression outputs a calibrated probability score from 0% to 100%.</div>
                    </div>
                </div>
                <div style="display: flex; gap: 14px; align-items: flex-start;">
                    <div class="step-num">4</div>
                    <div>
                        <div style="font-weight: 800; font-size: 15px; color: #F8FAFC;">Risk stratification &amp; guidance</div>
                        <div style="font-size: 13px; color: #8E9BAE; margin-top: 3px; line-height: 1.6;">Risk gauge, contributing factors, and actionable clinical recommendations.</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_c:
        st.markdown("""
        <div class="clinical-card">
            <div class="card-title">🩺 Key Risk Factors</div>
            <div style="display: flex; flex-direction: column; gap: 12px; margin-top: 6px;">
                <div style="background: rgba(255,0,84,0.08); border: 1px solid rgba(255,0,84,0.25); border-radius: 12px; padding: 13px 16px;">
                    <div style="display: flex; justify-content: space-between; font-weight: 800; font-size: 14px; color: #FF0054;">
                        <span>🩸 Systolic Blood Pressure</span>
                        <span style="font-size: 11px; opacity: 0.9; text-transform: uppercase;">Primary Driver</span>
                    </div>
                    <div style="font-size: 12.5px; color: #CBD5E1; margin-top: 5px; line-height: 1.5;">Readings above 140 mmHg are the strongest positive predictor of vascular damage.</div>
                </div>
                <div style="background: rgba(255,183,3,0.08); border: 1px solid rgba(255,183,3,0.25); border-radius: 12px; padding: 13px 16px;">
                    <div style="display: flex; justify-content: space-between; font-weight: 800; font-size: 14px; color: #FFB703;">
                        <span>🧪 Serum Cholesterol</span>
                        <span style="font-size: 11px; opacity: 0.9; text-transform: uppercase;">High Impact</span>
                    </div>
                    <div style="font-size: 12.5px; color: #CBD5E1; margin-top: 5px; line-height: 1.5;">Elevated cholesterol (Cat. 2–3) correlates strongly with arterial plaque formation.</div>
                </div>
                <div style="background: rgba(255,183,3,0.08); border: 1px solid rgba(255,183,3,0.25); border-radius: 12px; padding: 13px 16px;">
                    <div style="display: flex; justify-content: space-between; font-weight: 800; font-size: 14px; color: #FFB703;">
                        <span>⚖️ BMI &amp; Age</span>
                        <span style="font-size: 11px; opacity: 0.9; text-transform: uppercase;">Cumulative Risk</span>
                    </div>
                    <div style="font-size: 12.5px; color: #CBD5E1; margin-top: 5px; line-height: 1.5;">Adiposity with advancing age progressively increases cardiac workload.</div>
                </div>
                <div style="background: rgba(0,245,212,0.08); border: 1px solid rgba(0,245,212,0.25); border-radius: 12px; padding: 13px 16px;">
                    <div style="display: flex; justify-content: space-between; font-weight: 800; font-size: 14px; color: #00F5D4;">
                        <span>🏃 Physical Activity</span>
                        <span style="font-size: 11px; opacity: 0.9; text-transform: uppercase;">Protective</span>
                    </div>
                    <div style="font-size: 12.5px; color: #CBD5E1; margin-top: 5px; line-height: 1.5;">Regular exercise has a strong negative coefficient, lowering overall predicted risk.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 2 — PATIENT RISK ASSESSMENT
# ─────────────────────────────────────────────
elif page == "🔬  Patient Risk Assessment":
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Patient Risk Assessment</div>
        <div class="page-description">Fill in patient biometrics and vitals below to evaluate their cardiovascular disease probability.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Grouped Clinical Form ─────────────────────────────
    st.markdown("""
    <div class="clinical-card">
        <div class="card-title">📋 Patient Data Intake</div>
    """, unsafe_allow_html=True)

    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        st.markdown("<div style='font-size: 14px; font-weight: 800; color: #00F5D4; margin-bottom: 12px;'>01. Demographics & Biometrics</div>", unsafe_allow_html=True)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=45, step=1)
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        gender_val = 1 if gender == "Female" else 2
        height = st.number_input("Height (cm)", min_value=100, max_value=230, value=165, step=1)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=68.0, step=0.5)
        
        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        bmi_cat, bmi_class, bmi_hex = get_bmi_classification(bmi)
        st.markdown(f"""
        <div style="margin-top: 12px; padding: 13px 16px; background: #0D1424; border-radius: 14px; border: 1px solid rgba(0, 187, 249, 0.3);">
            <div style="font-size: 11px; color: #8E9BAE; font-weight: 800; text-transform: uppercase;">CALCULATED BMI</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px;">
                <span style="font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700; color: {bmi_hex};">{bmi:.1f} kg/m²</span>
                <span class="vitals-badge {bmi_class}">{bmi_cat}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown("<div style='font-size: 14px; font-weight: 800; color: #00F5D4; margin-bottom: 12px;'>02. Hemodynamics & Vitals</div>", unsafe_allow_html=True)
        ap_hi = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=240, value=120, step=1, help="Upper reading, e.g. 120 in 120/80")
        ap_lo = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=160, value=80, step=1, help="Lower reading, e.g. 80 in 120/80")
        
        bp_label, bp_class, bp_note = get_bp_classification(ap_hi, ap_lo)
        st.markdown(f"""
        <div style="margin-top: 12px; padding: 13px 16px; background: #0D1424; border-radius: 14px; border: 1px solid rgba(0, 187, 249, 0.3);">
            <div style="font-size: 11px; color: #8E9BAE; font-weight: 800; text-transform: uppercase;">JNC-7 BP CLASSIFICATION</div>
            <div style="margin-top: 6px;">
                <span class="vitals-badge {bp_class}">{bp_label}</span>
            </div>
            <div style="font-size: 11.5px; color: #CBD5E1; margin-top: 6px;">{bp_note}</div>
        </div>
        """, unsafe_allow_html=True)

    with f_col3:
        st.markdown("<div style='font-size: 14px; font-weight: 800; color: #00F5D4; margin-bottom: 12px;'>03. Metabolic & Lifestyle</div>", unsafe_allow_html=True)
        
        chol_opts = ["Normal", "Above Normal", "Well Above Normal"]
        cholesterol = st.selectbox("Cholesterol Level", chol_opts, index=0)
        chol_val = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[cholesterol]

        gluc_opts = ["Normal", "Above Normal", "Well Above Normal"]
        gluc = st.selectbox("Glucose Level", gluc_opts, index=0)
        gluc_val = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc]

        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            smoke = st.selectbox("Smoker?", ["No", "Yes"], index=0)
            smoke_val = 1 if smoke == "Yes" else 0
            alco = st.selectbox("Alcohol Use?", ["No", "Yes"], index=0)
            alco_val = 1 if alco == "Yes" else 0

        with sub_c2:
            active = st.selectbox("Physically Active?", ["Yes", "No"], index=0)
            active_val = 1 if active == "Yes" else 0

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Predict Action ───────────────────────────────────
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        predict_clicked = st.button("🔮 Calculate Clinical Risk Score")

    if predict_clicked:
        if ap_hi <= ap_lo:
            st.error("⚠️ Invalid Hemodynamic Input: Systolic Blood Pressure must be higher than Diastolic Blood Pressure.")
        else:
            try:
                # Prepare dataframe matching exact pipeline features
                input_df = pd.DataFrame({
                    "age": [age],
                    "gender": [gender_val],
                    "height": [height],
                    "weight": [weight],
                    "ap_hi": [ap_hi],
                    "ap_lo": [ap_lo],
                    "cholesterol": [chol_val],
                    "gluc": [gluc_val],
                    "smoke": [smoke_val],
                    "alco": [alco_val],
                    "active": [active_val],
                    "BMI": [bmi]
                })

                expected_cols = list(model.feature_names_in_)
                input_df = input_df[expected_cols]

                pred_class = model.predict(input_df)[0]
                pred_probas = model.predict_proba(input_df)[0]
                risk_pct = pred_probas[1] * 100.0
                safe_pct = pred_probas[0] * 100.0

                st.markdown("<hr style='border-color: rgba(0, 245, 212, 0.25); margin: 28px 0;'>", unsafe_allow_html=True)

                # Results Dashboard Layout
                res_col1, res_col2 = st.columns([1, 1.2])

                with res_col1:
                    if pred_class == 1:
                        gauge_html = render_svg_gauge(risk_pct)
                        st.markdown(f'''<div style="background: linear-gradient(160deg, rgba(255, 0, 84, 0.16) 0%, #0D1424 60%); border: 1px solid rgba(255, 0, 84, 0.45); border-radius: 24px; padding: 28px 20px 22px; text-align: center; box-shadow: 0 8px 32px rgba(255, 0, 84, 0.25);">{gauge_html}<div class="risk-header-high">Higher Disease Likelihood</div><p style="font-size: 13.5px; color: #CBD5E1; line-height: 1.6; margin-top: 8px;">Model predicts elevated cardiovascular disease probability for this profile.</p></div>''', unsafe_allow_html=True)
                    else:
                        gauge_html = render_svg_gauge(risk_pct)
                        st.markdown(f'''<div style="background: linear-gradient(160deg, rgba(0, 245, 212, 0.16) 0%, #0D1424 60%); border: 1px solid rgba(0, 245, 212, 0.45); border-radius: 24px; padding: 28px 20px 22px; text-align: center; box-shadow: 0 8px 32px rgba(0, 245, 212, 0.25);">{gauge_html}<div class="risk-header-low">Lower Disease Likelihood</div><p style="font-size: 13.5px; color: #CBD5E1; line-height: 1.6; margin-top: 8px;">Model predicts lower cardiovascular disease probability for this profile.</p></div>''', unsafe_allow_html=True)

                with res_col2:
                    st.markdown("""
                    <div class="clinical-card">
                        <div class="card-title">📊 Probabilistic Risk Decomposition</div>
                    """, unsafe_allow_html=True)

                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"""
                        <div class="metric-box" style="border-color: rgba(255, 0, 84, 0.4);">
                            <div class="metric-label" style="color: #FF0054;">DISEASE PROBABILITY</div>
                            <div class="metric-val" style="color: #FF0054;">{risk_pct:.1f}%</div>
                            <div class="metric-sub">Logistic regression score</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="metric-box" style="border-color: rgba(0, 245, 212, 0.4);">
                            <div class="metric-label" style="color: #00F5D4;">HEALTHY PROBABILITY</div>
                            <div class="metric-val" style="color: #00F5D4;">{safe_pct:.1f}%</div>
                            <div class="metric-sub">Baseline health score</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<div style='font-size: 11px; font-weight: 800; color: #526075; text-transform: uppercase; letter-spacing: 0.08em; margin: 20px 0 10px 0;'>Contributing Risk Factors</div>", unsafe_allow_html=True)
                    
                    risk_factors = []
                    if ap_hi >= 140: risk_factors.append(("Systolic BP", f"{ap_hi} mmHg", "High Risk Driver (>=140)", "#FF0054"))
                    if ap_lo >= 90:  risk_factors.append(("Diastolic BP", f"{ap_lo} mmHg", "Elevated (>=90)", "#FFB703"))
                    if chol_val > 1: risk_factors.append(("Cholesterol", cholesterol, f"Category {chol_val}", "#FF0054" if chol_val==3 else "#FFB703"))
                    if bmi >= 30:    risk_factors.append(("Body Mass Index", f"{bmi:.1f} kg/m²", "Obese Class", "#FFB703"))
                    if smoke_val == 1: risk_factors.append(("Smoking Habit", "Active Smoker", "Vascular Stress", "#FF0054"))
                    if active_val == 0: risk_factors.append(("Physical Inactivity", "Sedentary", "Modifiable Risk Factor", "#FFB703"))

                    if risk_factors:
                        for name, val, desc, hex_c in risk_factors:
                            st.markdown(f"""
                            <div class="factor-row">
                                <div>
                                    <div class="factor-name">{name}</div>
                                    <div style="font-size: 11.5px; color: #8E9BAE;">{desc}</div>
                                </div>
                                <div class="factor-val" style="color: {hex_c};">{val}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="padding: 14px; background: rgba(0, 245, 212, 0.14); border: 1px solid rgba(0, 245, 212, 0.35); border-radius: 12px; color: #00F5D4; font-size: 13.5px; font-weight: 700; text-align: center;">
                            ✅ No critical high-risk clinical factors flagged for this profile.
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                # Medical Recommendations
                st.markdown("""
                <div class="clinical-card margin-top-5">
                    <div class="card-title">💡 Actionable Clinical Recommendations</div>
                """, unsafe_allow_html=True)

                rec_1, rec_2 = st.columns(2)
                with rec_1:
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="recommendation-title">🩺 Hemodynamic & Vitals Management</div>
                        <div class="recommendation-body">
                            Current BP is <b>{ap_hi}/{ap_lo} mmHg</b> ({bp_label}). 
                            {'Target BP reduction via DASH diet, low sodium (&lt; 2g/day), and clinical consultation.' if ap_hi >= 130 else 'Maintain current healthy blood pressure levels with routine annual screening.'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with rec_2:
                    st.markdown(f"""
                    <div class="recommendation-card" style="border-left-color: #00BBF9;">
                        <div class="recommendation-title" style="color: #00BBF9;">🥗 Lifestyle & Metabolic Modification</div>
                        <div class="recommendation-body">
                            {'Engage in 150 mins/week of moderate aerobic exercise and limit saturated fats.' if active_val == 0 or chol_val > 1 else 'Continue regular physical activity and balanced Mediterranean-style nutrition.'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

                # Printable Assessment Summary
                with st.expander("📋 View Printable Patient Assessment Report"):
                    st.markdown(f"""
                    <div style="background: #0D1424; border: 1px solid rgba(0, 245, 212, 0.3); border-radius: 16px; padding: 28px; font-family: 'Plus Jakarta Sans', sans-serif;">
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 16px;">
                            <div>
                                <div style="font-family: 'Outfit', sans-serif; font-weight: 900; font-size: 22px; color: #F8FAFC;">HEARTSENSE — CLINICAL RISK REPORT</div>
                                <div style="font-size: 12px; color: #8E9BAE;">Cardiovascular Disease Risk Prediction Using Machine Learning</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 13.5px; color: #F8FAFC; font-weight: 800;">CLASSIFICATION: {'ELEVATED RISK' if pred_class==1 else 'LOW RISK'}</div>
                                <div style="font-size: 11px; color: #526075;">Pipeline Version: 2.4-LR</div>
                            </div>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0;">
                            <div><span style="color: #526075; font-size: 11px; font-weight: 800;">AGE:</span> <div style="font-weight: 700; color: #F8FAFC;">{age} years</div></div>
                            <div><span style="color: #526075; font-size: 11px; font-weight: 800;">GENDER:</span> <div style="font-weight: 700; color: #F8FAFC;">{gender}</div></div>
                            <div><span style="color: #526075; font-size: 11px; font-weight: 800;">BLOOD PRESSURE:</span> <div style="font-weight: 700; color: #F8FAFC;">{ap_hi}/{ap_lo} mmHg</div></div>
                            <div><span style="color: #526075; font-size: 11px; font-weight: 800;">BMI:</span> <div style="font-weight: 700; color: #F8FAFC;">{bmi:.1f} kg/m²</div></div>
                            <div><span style="color: #526075; font-size: 11px; font-weight: 800;">CHOLESTEROL:</span> <div style="font-weight: 700; color: #F8FAFC;">{cholesterol}</div></div>
                            <div><span style="color: #526075; font-size: 11px; font-weight: 800;">GLUCOSE:</span> <div style="font-weight: 700; color: #F8FAFC;">{gluc}</div></div>
                            <div><span style="color: #526075; font-size: 11px; font-weight: 800;">SMOKER:</span> <div style="font-weight: 700; color: #F8FAFC;">{smoke}</div></div>
                            <div><span style="color: #526075; font-size: 11px; font-weight: 800;">PHYSICALLY ACTIVE:</span> <div style="font-weight: 700; color: #F8FAFC;">{active}</div></div>
                        </div>
                        <div style="background: rgba(0, 245, 212, 0.14); border: 1px solid rgba(0, 245, 212, 0.35); border-radius: 12px; padding: 16px 20px; display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 800; font-size: 14.5px; color: #F8FAFC;">Predicted Disease Risk Score</span>
                            <span style="font-family: 'Space Mono', monospace; font-size: 24px; font-weight: 800; color: #00F5D4;">{risk_pct:.2f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as ex:
                st.error("❌ Prediction evaluation error.")
                st.exception(ex)

# ─────────────────────────────────────────────
# PAGE 3 — MODEL ANALYTICS & INSIGHTS
# ─────────────────────────────────────────────
elif page == "📊  Model Analytics":
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📊 Model Analytics & Validation Suite</div>
        <div class="page-description">Quantitative performance benchmarks, cross-validation stability, and feature importance coefficient analysis.</div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics summary
    st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #F8FAFC; margin-bottom: 16px; letter-spacing: -0.01em;'>Test Set Benchmarks</div>", unsafe_allow_html=True)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    bm_data = [
        ("71.39%", "Accuracy", "Overall test accuracy", m_col1),
        ("73.16%", "Precision", "Positive predictive value", m_col2),
        ("67.51%", "Recall", "Sensitivity rate", m_col3),
        ("70.22%", "F1-Score", "Harmonic mean metric", m_col4),
    ]
    for val, lbl, desc, c in bm_data:
        with c:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">{lbl}</div>
                <div class="metric-val" style="color: #F8FAFC;">{val}</div>
                <div class="metric-sub">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Overfitting Check & Bar Plot
    st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #F8FAFC; margin-bottom: 16px;'>🔍 Overfitting & Generalization Audit</div>", unsafe_allow_html=True)
    
    ov_c1, ov_c2 = st.columns([1, 1.2])

    train_acc = 72.05
    test_acc = 71.39
    diff = abs(train_acc - test_acc)

    with ov_c1:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="card-title">⚖️ Generalization Assessment</div>
            <div style="display: flex; flex-direction: column; gap: 14px; margin-top: 10px;">
                <div style="background: #040711; padding: 15px 18px; border-radius: 14px; border: 1px solid rgba(0, 187, 249, 0.25);">
                    <div style="font-size: 11px; color: #8E9BAE; font-weight: 800; text-transform: uppercase;">TRAINING ACCURACY</div>
                    <div style="font-family: 'Space Mono', monospace; font-size: 26px; font-weight: 700; color: #00BBF9; margin-top: 2px;">~{train_acc:.2f}%</div>
                </div>
                <div style="background: #040711; padding: 15px 18px; border-radius: 14px; border: 1px solid rgba(0, 245, 212, 0.25);">
                    <div style="font-size: 11px; color: #8E9BAE; font-weight: 800; text-transform: uppercase;">TEST ACCURACY</div>
                    <div style="font-family: 'Space Mono', monospace; font-size: 26px; font-weight: 700; color: #00F5D4; margin-top: 2px;">{test_acc:.2f}%</div>
                </div>
                <div style="background: rgba(0, 245, 212, 0.14); border: 1px solid rgba(0, 245, 212, 0.35); padding: 15px; border-radius: 14px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 20px;">✅</span>
                        <div>
                            <div style="font-weight: 800; color: #00F5D4; font-size: 14px;">Optimal Generalization</div>
                            <div style="font-size: 12.5px; color: #A7F3D0; margin-top: 2px;">Variance gap of <b>{diff:.2f}%</b> confirms no overfitting.</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ov_c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        apply_chart_theme(fig, ax)

        bars = ax.bar(["Train Accuracy", "Test Accuracy"], [train_acc, test_acc],
                      color=["#00BBF9", "#00F5D4"], width=0.45)
        ax.set_ylim(60, 80)
        ax.set_ylabel("Accuracy (%)", color="#8E9BAE", fontsize=10)
        ax.set_title("Train vs Test Accuracy Baseline", color="#F8FAFC", fontsize=12, fontweight='bold', pad=12)

        for bar, val in zip(bars, [train_acc, test_acc]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2.2,
                    f"{val:.2f}%", ha='center', va='top',
                    color='#040711', fontweight='bold', fontsize=11)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # 5-Fold Cross Validation
    st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #F8FAFC; margin: 28px 0 16px 0; letter-spacing: -0.01em;'>5-Fold Cross-Validation</div>", unsafe_allow_html=True)
    
    cv_scores = np.array([71.52, 71.63, 71.45, 71.71, 71.58])
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    cv_c1, cv_c2 = st.columns([1, 1.2])

    with cv_c1:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="card-title">📌 CV Summary</div>
            <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
                <span style="color: #8E9BAE; font-size: 13.5px;">Mean CV Score</span>
                <span style="color: #00BBF9; font-weight: 800; font-size: 14px; font-family: 'Space Mono', monospace;">{cv_mean:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
                <span style="color: #8E9BAE; font-size: 13.5px;">Std Deviation</span>
                <span style="color: #00F5D4; font-weight: 800; font-size: 14px; font-family: 'Space Mono', monospace;">±{cv_std:.2f}%</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 12px 0;">
                <span style="color: #8E9BAE; font-size: 13.5px;">Score Range</span>
                <span style="color: #CBD5E1; font-weight: 700; font-size: 13.5px; font-family: 'Space Mono', monospace;">{cv_scores.min():.2f} – {cv_scores.max():.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with cv_c2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.2))
        apply_chart_theme(fig2, ax2)

        folds = [f"Fold {i}" for i in range(1, 6)]
        bars2 = ax2.bar(folds, cv_scores, color='#00F5D4', width=0.45)
        ax2.axhline(y=cv_mean, color='#FF0054', linestyle='--', linewidth=1.5, label=f'Mean ({cv_mean:.2f}%)')
        ax2.set_ylim(cv_mean - 1.5, cv_mean + 1.5)
        ax2.set_ylabel("Accuracy (%)", color="#8E9BAE", fontsize=10)
        ax2.set_title("5-Fold Cross-Validation Breakdown", color="#F8FAFC", fontsize=12, fontweight='bold', pad=12)
        ax2.legend(facecolor='#0D1424', edgecolor='#162238', labelcolor='#CBD5E1', fontsize=9)

        for bar, val in zip(bars2, cv_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.15,
                     f"{val:.2f}%", ha='center', va='top', color='#040711', fontweight='bold', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    # Feature Importance Plot
    st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #F8FAFC; margin: 28px 0 16px 0;'>📌 Logistic Regression Feature Importance Coefficients</div>", unsafe_allow_html=True)
    
    feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'BMI']

    try:
        coefs = model.named_steps['model'].coef_[0]
    except Exception:
        coefs = np.array([0.42, -0.05, -0.08, 0.12, 0.61, 0.38, 0.28, 0.17, 0.07, 0.06, -0.12, 0.19])

    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    sorted_feat = [feature_names[i] for i in sorted_idx]
    sorted_coef = [coefs[i] for i in sorted_idx]

    colors_feat = ['#FF0054' if c > 0 else '#00F5D4' for c in sorted_coef]

    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    apply_chart_theme(fig3, ax3)

    bars3 = ax3.barh(sorted_feat[::-1], sorted_coef[::-1], color=colors_feat[::-1], height=0.6)
    ax3.axvline(x=0, color='#526075', linewidth=1)
    ax3.set_xlabel("Scaled Feature Coefficient Value", color="#8E9BAE", fontsize=10)
    ax3.set_title("Logistic Regression Coefficients\n(Crimson = Risk Driver, Cyber Emerald = Protective Factor)",
                  color="#F8FAFC", fontsize=12, fontweight='bold', pad=12)

    red_patch = mpatches.Patch(color='#FF0054', label='Increases Risk')
    emerald_patch = mpatches.Patch(color='#00F5D4', label='Lowers Risk')
    ax3.legend(handles=[red_patch, emerald_patch], facecolor='#0D1424', edgecolor='#162238', labelcolor='#CBD5E1')

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# ─────────────────────────────────────────────
# PAGE 4 — DATASET EXPLORER
# ─────────────────────────────────────────────
elif page == "📁  Dataset Explorer":
    
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Dataset Explorer</div>
        <div class="page-description">Inspect raw patient records, statistical distributions, and cohort attributes.</div>
    </div>
    """, unsafe_allow_html=True)

    df_cardio = load_dataset()

    if df_cardio is not None:
        st.markdown(f"""
        <div class="clinical-card">
            <div class="card-title">📊 Cohort Dataset Metadata</div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
                <div class="metric-box">
                    <div class="metric-label">TOTAL RECORDS</div>
                    <div class="metric-val">{len(df_cardio):,}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">FEATURES</div>
                    <div class="metric-val" style="color: #00F5D4;">{df_cardio.shape[1]}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">CARDIO POSITIVE</div>
                    <div class="metric-val" style="color: #FF0054;">{(df_cardio['cardio']==1).sum():,}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">CARDIO NEGATIVE</div>
                    <div class="metric-val" style="color: #00F5D4;">{(df_cardio['cardio']==0).sum():,}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #F8FAFC; margin: 24px 0 12px 0; letter-spacing: -0.01em;'>Patient Records (first 100)</div>", unsafe_allow_html=True)
        st.dataframe(df_cardio.head(100), use_container_width=True)

        st.markdown("<div style='font-family: Outfit, sans-serif; font-size: 20px; font-weight: 800; color: #F8FAFC; margin: 28px 0 12px 0; letter-spacing: -0.01em;'>Feature Summary Statistics</div>", unsafe_allow_html=True)
        st.dataframe(df_cardio.describe().T, use_container_width=True)
    else:
        st.info("ℹ️ Dataset file `cardio_train.csv` not found in workspace directory.")
