
import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="🌸 Iris AI Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="hero-section">
    <p class="hero-title">🌸 Iris AI Studio</p>
    <p class="hero-subtitle">Production-Grade Flower Classification · SVM · 96.67% Accuracy</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Best Accuracy", "96.67%", "+6.67% above baseline")
col2.metric("🤖 Models Trained", "5", "LR, KNN, DT, RF, SVM")
col3.metric("📦 Dataset Size", "150 rows", "4 features · 3 classes")
col4.metric("⚡ Inference Time", "<10ms", "Real-time prediction")

st.markdown("---")
st.markdown("### 🚀 Navigate using the sidebar to explore all features")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""<div class="glass-card">
        <h4>📊 Data Explorer</h4>
        <p>Explore the Iris dataset with interactive charts and statistics.</p>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""<div class="glass-card">
        <h4>🔮 Live Prediction</h4>
        <p>Enter flower measurements and get instant AI predictions.</p>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""<div class="glass-card">
        <h4>📋 Batch Predict</h4>
        <p>Upload a CSV and classify hundreds of flowers at once.</p>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown("""<div class="glass-card">
        <h4>🧠 Explain AI</h4>
        <p>Understand model decisions using SHAP explainability.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem;">
    Built by <b>Akshaya Vasireddi</b> · Iris AI Studio v1.0 · Powered by Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)
