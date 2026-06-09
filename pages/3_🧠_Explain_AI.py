
import streamlit as st
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import load_model, load_data

st.set_page_config(page_title="Explain AI", page_icon="🧠", layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🧠 Explainable AI (SHAP)")
st.markdown("Understand **why** the model makes each prediction.")

model, scaler, encoder = load_model()
df = load_data()

X = df.drop("species", axis=1)
X_scaled = scaler.transform(X)

with st.spinner("Computing SHAP values..."):
    explainer   = shap.KernelExplainer(model.predict_proba, shap.sample(X_scaled, 30))
    shap_values = explainer.shap_values(X_scaled[:50])

st.subheader("🌍 Global Feature Importance")
st.markdown("Average impact of each feature across all predictions.")

feature_names = list(X.columns)
mean_shap = np.abs(np.array(shap_values)).mean(axis=(0, 2)) if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)

import plotly.express as px
importance_df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_shap})
importance_df = importance_df.sort_values("Mean |SHAP|", ascending=True)
fig = px.bar(importance_df, x="Mean |SHAP|", y="Feature", orientation="h",
             template="plotly_dark", color="Mean |SHAP|",
             color_continuous_scale="Viridis")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔍 Local Explanation — Single Prediction")
sample_idx = st.slider("Select a sample row to explain", 0, 49, 0)
st.markdown(f"**Sample #{sample_idx}** — Actual: `{df['species'].iloc[sample_idx]}`")

fig2, ax = plt.subplots()
shap.waterfall_plot(
    shap.Explanation(
        values        = shap_values[0][sample_idx] if isinstance(shap_values, list) else shap_values[sample_idx],
        base_values   = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        data          = X_scaled[sample_idx],
        feature_names = feature_names
    ), show=False
)
st.pyplot(plt.gcf())
plt.clf()

st.info("💡 Red bars push the prediction **higher**, blue bars push it **lower**.")
