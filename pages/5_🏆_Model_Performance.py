
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import load_model, load_data

st.set_page_config(page_title="Model Performance", page_icon="🏆", layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🏆 Model Performance")

model, scaler, encoder = load_model()
df = load_data()
le = LabelEncoder()
y  = le.fit_transform(df["species"])
X_scaled = scaler.transform(df.drop("species", axis=1))
y_pred   = model.predict(X_scaled)

# ── Metrics Table ──
results = pd.DataFrame({
    "Model":        ["SVM ⭐","Logistic Regression","KNN","Decision Tree","Random Forest"],
    "Accuracy (%)": [96.67, 93.33, 93.33, 90.00, 90.00],
    "Precision (%)": [96.97, 93.33, 94.44, 90.24, 90.24],
    "Recall (%)":   [96.67, 93.33, 93.33, 90.00, 90.00],
    "F1 Score (%)": [96.66, 93.33, 93.27, 89.97, 89.97],
})

st.subheader("📋 All Models Comparison")
st.dataframe(results, use_container_width=True)

fig = px.bar(results, x="Model", y="Accuracy (%)", color="Accuracy (%)",
             template="plotly_dark", color_continuous_scale="Viridis",
             text="Accuracy (%)")
fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Confusion Matrix ──
st.subheader("🔲 Confusion Matrix (SVM)")
cm = confusion_matrix(y, y_pred)
fig2 = ff.create_annotated_heatmap(
    cm.tolist(), x=list(le.classes_), y=list(le.classes_),
    colorscale="Blues", showscale=True
)
fig2.update_layout(
    xaxis_title="Predicted", yaxis_title="Actual",
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Classification Report ──
st.subheader("📊 Classification Report")
report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)
st.dataframe(report_df, use_container_width=True)
