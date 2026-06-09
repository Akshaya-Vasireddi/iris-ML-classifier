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
# Load CSS
# =========================
css_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "style.css"
)

if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# =========================
# Header
# =========================
st.title("🧠 Explainable AI (SHAP)")
st.markdown(
    "Understand **why** the machine learning model makes each prediction."
)

# =========================
# Load Data & Model
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

shap_arr = np.array(shap_values)

# Debug information
st.write("SHAP Output Shape:", shap_arr.shape)

try:

    # SHAP returns (samples, features, classes)
    if shap_arr.ndim == 3:

        mean_shap = np.abs(shap_arr).mean(axis=(0, 2))

    # SHAP returns (samples, features)
    elif shap_arr.ndim == 2:

        mean_shap = np.abs(shap_arr).mean(axis=0)

    # SHAP returns list[class]
    elif isinstance(shap_values, list):

        mean_shap = np.mean(
            [np.abs(v).mean(axis=0) for v in shap_values],
            axis=0
        )

    else:

        st.error(f"Unsupported SHAP shape: {shap_arr.shape}")
        st.stop()

    mean_shap = np.array(mean_shap).flatten()

    # Fix mismatch automatically
    if len(mean_shap) > len(feature_names):
        mean_shap = mean_shap[:len(feature_names)]

    if len(mean_shap) < len(feature_names):
        st.error(
            f"Feature mismatch. Features={len(feature_names)}, "
            f"SHAP={len(mean_shap)}"
        )
        st.stop()

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
        color="Mean |SHAP|",
        color_continuous_scale="Viridis",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except Exception as e:
    st.error(f"Global SHAP error: {e}")

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

try:

    if isinstance(shap_values, list):

        values = shap_values[0][sample_idx]
        base_value = explainer.expected_value[0]

    elif shap_arr.ndim == 3:

        values = shap_arr[sample_idx, :, 0]
        base_value = (
            explainer.expected_value[0]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

    else:

        values = shap_arr[sample_idx]
        base_value = explainer.expected_value

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
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

except Exception as e:
    st.error(f"Local SHAP error: {e}")

# =========================
# Information
# =========================
st.info(
    "💡 Red bars increase the prediction score, while blue bars decrease it."
)
