import random
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------------------
# Generate a field of twinkling stars for the background
# ---------------------------------------------------------------------
def generate_stars(count: int = 70) -> str:
    stars_html = ""
    for _ in range(count):
        top = random.uniform(0, 100)
        left = random.uniform(0, 100)
        size = random.uniform(1, 2.5)
        delay = random.uniform(0, 6)
        duration = random.uniform(3, 7)
        stars_html += (
            f'<div class="star" style="'
            f'top:{top}%; left:{left}%; width:{size}px; height:{size}px; '
            f'animation-delay:{delay}s; animation-duration:{duration}s;"></div>'
        )
    return stars_html


STARFIELD = generate_stars(70)


# ---------------------------------------------------------------------
# Theme / styling
# ---------------------------------------------------------------------
st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: radial-gradient(ellipse at top, #0f1420 0%, #0b0f19 55%, #070a10 100%);
    color: #f5f7fa;
    overflow: hidden;
}}

/* Starfield sits behind content, kept subtle so text stays sharp */
.star-field {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}}

.star {{
    position: absolute;
    background: #ffffff;
    border-radius: 50%;
    opacity: 0.12;
    animation: twinkle ease-in-out infinite;
}}

@keyframes twinkle {{
    0%, 100% {{ opacity: 0.08; transform: scale(1); }}
    50% {{ opacity: 0.55; transform: scale(1.5); }}
}}

.orb {{
    position: fixed;
    border-radius: 50%;
    filter: blur(100px);
    z-index: 0;
    pointer-events: none;
    opacity: 0.5;
}}

.orb-1 {{
    width: 360px;
    height: 360px;
    top: -140px;
    left: -100px;
    background: rgba(249, 115, 22, 0.14);
    animation: drift 20s ease-in-out infinite;
}}

.orb-2 {{
    width: 300px;
    height: 300px;
    bottom: -120px;
    right: -80px;
    background: rgba(37, 99, 235, 0.16);
    animation: drift 24s ease-in-out infinite reverse;
}}

@keyframes drift {{
    0%, 100% {{ transform: translate(0, 0); }}
    50% {{ transform: translate(30px, 25px); }}
}}

.block-container {{
    position: relative;
    z-index: 1;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: rgba(17, 24, 39, 0.92);
    border-right: 1px solid #263041;
    backdrop-filter: blur(6px);
}}

/* Main Title */
.main-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 42px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 5px;
}}

.subtitle {{
    font-size: 16px;
    color: #9ca3af;
    margin-bottom: 30px;
    max-width: 720px;
    line-height: 1.6;
}}

/* Cards */
.card {{
    background: linear-gradient(145deg, #151c2b, #101622);
    border: 1px solid #273244;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    transition: border-color 0.25s ease, transform 0.25s ease;
}}

.card:hover {{
    border-color: #f97316;
    transform: translateY(-2px);
}}

.card-title {{
    color: #ffffff;
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 6px;
}}

.card-text {{
    color: #9ca3af;
    font-size: 14px;
    line-height: 1.6;
}}

/* Result card */
.result-card {{
    background: linear-gradient(135deg, #1c2a52, #111827);
    border: 1px solid #2563eb;
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 10px 35px rgba(37, 99, 235, 0.2);
}}

.cluster-number {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 52px;
    font-weight: 800;
    color: #60a5fa;
    margin: 6px 0;
}}

.cluster-label {{
    font-size: 16px;
    color: #d1d5db;
}}

/* Metrics */
div[data-testid="stMetric"] {{
    background: #151c2b;
    border: 1px solid #273244;
    padding: 18px;
    border-radius: 14px;
    transition: border-color 0.25s ease;
}}

div[data-testid="stMetric"]:hover {{
    border-color: #f97316;
}}

div[data-testid="stMetricLabel"] {{
    color: #9ca3af;
}}

div[data-testid="stMetricValue"] {{
    color: #ffffff;
    font-weight: 700;
}}

/* Buttons */
.stButton > button {{
    width: 100%;
    height: 48px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(90deg, #f97316, #ea580c);
    color: white;
    font-size: 16px;
    font-weight: 700;
    transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
    box-shadow: 0 4px 18px rgba(249, 115, 22, 0.25);
}}

.stButton > button:hover {{
    filter: brightness(1.08);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(249, 115, 22, 0.35);
}}

.stButton > button:active {{
    transform: translateY(0);
}}

/* Inputs */
div[data-baseweb="input"] {{
    background: #111827;
    border-radius: 10px;
    border: 1px solid #273244;
}}

div[data-baseweb="select"] {{
    background: #111827;
    border-radius: 10px;
    border: 1px solid #273244;
}}

div[data-testid="stSlider"] {{
    color: #f97316;
}}

/* Hide Streamlit branding */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header {{ visibility: hidden; }}

</style>

<div class="star-field">{STARFIELD}</div>
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Load trained model
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features


kmeans, scaler, features = load_model()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:

    st.markdown("<h2 style='color:white;'>Customer Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9ca3af;'>K-Means Customer Segmentation</p>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<p style='color:#9ca3af; letter-spacing:0.5px;'>MODEL INFORMATION</p>", unsafe_allow_html=True)

    st.write("Algorithm")
    st.markdown("<b style='color:#60a5fa;'>K-Means Clustering</b>", unsafe_allow_html=True)

    st.write("Features")

    for feature in features:
        st.markdown(f"<span style='color:#d1d5db;'>• {feature}</span>", unsafe_allow_html=True)

    st.write("")

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Model Status</div>
            <div class="card-text">Trained model loaded successfully.</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown('<div class="main-title">Customer Segmentation</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">'
    'Analyze customer behavior and identify meaningful customer segments '
    'using K-Means clustering.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Algorithm", "K-Means")

with col2:
    st.metric("Clusters", kmeans.n_clusters)

with col3:
    st.metric("Input Features", len(features))

with col4:
    st.metric("Scaling", "StandardScaler")

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Customer input section
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="card">
        <div class="card-title">Customer Profile</div>
        <div class="card-text">
            Enter customer information to predict which segment the customer belongs to.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)

with col2:
    income = st.number_input("Annual Income (k$)", min_value=1, max_value=500, value=50, step=1)

with col3:
    spending = st.slider("Spending Score", min_value=1, max_value=100, value=50)

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
predict_button = st.button("Predict Customer Segment")

if predict_button:

    customer_data = pd.DataFrame([[age, income, spending]], columns=features)
    customer_scaled = scaler.transform(customer_data)
    cluster = kmeans.predict(customer_scaled)[0]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="cluster-label">Predicted Customer Segment</div>
            <div class="cluster-number">Cluster {cluster}</div>
            <div class="cluster-label">Customer successfully assigned to this segment</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Customer Summary</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Age", f"{age} years")

    with col2:
        st.metric("Annual Income", f"${income}k")

    with col3:
        st.metric("Spending Score", spending)

    distances = kmeans.transform(customer_scaled)[0]

    distance_df = pd.DataFrame({
        "Cluster": [f"Cluster {i}" for i in range(len(distances))],
        "Distance": distances
    })

    distance_df = distance_df.sort_values("Distance")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
            <div class="card-title">Cluster Confidence Analysis</div>
            <div class="card-text">
                Distance from each cluster centroid. Lower distance means the
                customer is more similar to that cluster.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    fig = px.bar(
        distance_df,
        x="Cluster",
        y="Distance",
        text_auto=".2f"
    )

    fig.update_traces(marker_color="#f97316")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#ffffff"),
        xaxis_title="Customer Segment",
        yaxis_title="Distance from Centroid",
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown(
    """
    <br><br>
    <div style="
        text-align:center;
        color:#6b7280;
        padding:20px;
        border-top:1px solid #273244;
    ">
        Customer Segmentation Dashboard
        <br>
        Built with Streamlit • K-Means • Scikit-Learn
    </div>
    """,
    unsafe_allow_html=True
)