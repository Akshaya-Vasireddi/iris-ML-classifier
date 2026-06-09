
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import load_model

st.set_page_config(page_title="Batch Prediction", page_icon="📋", layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📋 Batch Prediction")
st.markdown("Upload a CSV file with flower measurements to classify multiple samples at once.")

st.info("📌 CSV must have columns: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`")

# Sample download
sample = pd.DataFrame({
    "sepal_length": [5.1, 6.2, 7.3],
    "sepal_width":  [3.5, 2.9, 3.0],
    "petal_length": [1.4, 4.3, 6.3],
    "petal_width":  [0.2, 1.3, 1.8],
})
st.download_button("⬇️ Download Sample CSV", sample.to_csv(index=False),
                   "sample_iris.csv", "text/csv")

uploaded = st.file_uploader("📁 Upload your CSV", type=["csv"])

if uploaded:
    df_input = pd.read_csv(uploaded)
    st.subheader("📄 Uploaded Data")
    st.dataframe(df_input, use_container_width=True)

    required = ["sepal_length","sepal_width","petal_length","petal_width"]
    if all(c in df_input.columns for c in required):
        model, scaler, encoder = load_model()
        X = df_input[required].values
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled).max(axis=1) * 100

        df_input["Predicted Species"] = encoder.inverse_transform(preds)
        df_input["Confidence (%)"]    = probs.round(2)

        st.subheader("✅ Prediction Results")
        st.dataframe(df_input, use_container_width=True)

        # Summary
        st.subheader("📊 Summary")
        summary = df_input["Predicted Species"].value_counts().reset_index()
        summary.columns = ["Species", "Count"]
        st.dataframe(summary, use_container_width=True)

        st.download_button("⬇️ Download Results CSV",
                           df_input.to_csv(index=False),
                           "iris_predictions.csv", "text/csv")
    else:
        st.error(f"❌ Missing columns! Required: {required}")
