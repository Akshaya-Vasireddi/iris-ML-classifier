import streamlit as st
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.helper import load_model, load_data

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="🧠 Explain AI",
    page_icon="🧠",
    layout="wide"
)

# =========================
# Load CSS Safely
# =========================
css_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "style.css"
)

if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.title("🧠 Explainable AI (SHAP)")
st.markdown(
    "Understand **why** the machine learning model makes each prediction."
)

# =========================
# Load Model & Data
# =========================
model, scaler, encoder = load_model()
df = load_data()

X = df.drop("species", axis=1)
X_scaled = scaler.transform(X)

# =========================
# SHAP Computation
# =========================
with st.spinner("Computing SHAP values..."):

    background = shap.sample(X_scaled, 30)

    explainer = shap.KernelExplainer(
        model.predict_proba,
        background
    )

    shap_values = explainer.shap_values(X_scaled[:50])

# =========================
# Global Feature Importance
# =========================
st.subheader("🌍 Global Feature Importance")
st.markdown(
    "Average impact of each feature across all model predictions."
)

feature_names = list(X.columns)

if isinstance(shap_values, list):
    mean_shap = np.mean(
        [np.abs(class_vals).mean(axis=0) for class_vals in shap_values],
        axis=0
    )
else:
    mean_shap = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean |SHAP|": mean_shap
})

importance_df = importance_df.sort_values(
    "Mean |SHAP|",
    ascending=True
)

fig = px.bar(
    importance_df,
    x="Mean |SHAP|",
    y="Feature",
    orientation="h",
    template="plotly_dark",
    color="Mean |SHAP|",
    color_continuous_scale="Viridis",
    title="Feature Importance"
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# Local Explanation
# =========================
st.subheader("🔍 Local Explanation")

sample_idx = st.slider(
    "Select a sample row",
    min_value=0,
    max_value=min(49, len(df) - 1),
    value=0
)

st.markdown(
    f"**Sample #{sample_idx}** — Actual Class: "
    f"`{df['species'].iloc[sample_idx]}`"
)

if isinstance(shap_values, list):

    explanation = shap.Explanation(
        values=shap_values[0][sample_idx],
        base_values=explainer.expected_value[0],
        data=X_scaled[sample_idx],
        feature_names=feature_names
    )

else:

    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_scaled[sample_idx],
        feature_names=feature_names
    )

fig2 = plt.figure(figsize=(10, 6))

shap.waterfall_plot(
    explanation,
    show=False
)

st.pyplot(fig2)

plt.close()

# =========================
# Info Box
# =========================
st.info(
    "💡 Red bars increase the prediction score, "
    "while blue bars decrease it."
)
