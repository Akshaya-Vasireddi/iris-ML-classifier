
import streamlit as st
import pandas as pd
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import load_data

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Data Explorer")
st.markdown("Interactively explore the Iris dataset.")

df = load_data()

# ── Overview ──
st.subheader("👀 Dataset Preview")
st.dataframe(df, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Total Samples", df.shape[0])
c2.metric("Features", df.shape[1] - 1)
c3.metric("Species", df["species"].nunique())

st.markdown("---")

# ── Distribution ──
st.subheader("📈 Feature Distribution")
feature = st.selectbox("Select Feature", ["sepal_length","sepal_width","petal_length","petal_width"])
fig = px.histogram(df, x=feature, color="species", barmode="overlay",
                   nbins=25, template="plotly_dark",
                   color_discrete_sequence=["#667eea","#f093fb","#4facfe"])
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Scatter ──
st.subheader("🔵 Interactive Scatter Plot")
col1, col2 = st.columns(2)
x_ax = col1.selectbox("X Axis", ["sepal_length","sepal_width","petal_length","petal_width"], index=2)
y_ax = col2.selectbox("Y Axis", ["sepal_length","sepal_width","petal_length","petal_width"], index=3)
fig2 = px.scatter(df, x=x_ax, y=y_ax, color="species", size_max=10,
                  template="plotly_dark", symbol="species",
                  color_discrete_sequence=["#667eea","#f093fb","#4facfe"])
fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Correlation Heatmap ──
st.subheader("🔥 Correlation Heatmap")
corr = df.drop("species", axis=1).corr().round(2)
fig3 = px.imshow(corr, text_auto=True, template="plotly_dark",
                 color_continuous_scale="RdBu_r")
fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# ── Boxplot ──
st.subheader("📦 Boxplot by Species")
feat_box = st.selectbox("Feature for Boxplot", ["sepal_length","sepal_width","petal_length","petal_width"], key="box")
fig4 = px.box(df, x="species", y=feat_box, color="species", template="plotly_dark",
              color_discrete_sequence=["#667eea","#f093fb","#4facfe"], points="all")
fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig4, use_container_width=True)
