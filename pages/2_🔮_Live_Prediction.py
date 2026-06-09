
import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helper import load_model, predict_flower, SPECIES_INFO

st.set_page_config(page_title="Live Prediction", page_icon="🔮", layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🔮 Live Prediction")
st.markdown("Adjust the sliders and click **Predict** to classify your Iris flower in real time.")

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📐 Enter Measurements")
    sepal_length = st.slider("🌿 Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
    sepal_width  = st.slider("🌿 Sepal Width (cm)",  2.0, 4.5, 3.5, 0.1)
    petal_length = st.slider("🌸 Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
    petal_width  = st.slider("🌸 Petal Width (cm)",  0.1, 2.5, 0.2, 0.1)
    predict_btn  = st.button("🚀 Predict Species", use_container_width=True)

with col2:
    if predict_btn:
        with st.spinner("Classifying..."):
            features = [sepal_length, sepal_width, petal_length, petal_width]
            species, confidence, probs, classes = predict_flower(features)
            info = SPECIES_INFO[species]

        st.markdown(f"""
        <div class="prediction-box">
            <h2>{info["emoji"]} {species}</h2>
            <p style="font-size:1.4rem; font-weight:700;">Confidence: {confidence:.1f}%</p>
            <p>{info["description"]}</p>
            <p><i>💡 {info["fact"]}</i></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📊 Probability Breakdown")
        fig = go.Figure(go.Bar(
            x=list(classes),
            y=[round(p*100, 2) for p in probs],
            marker_color=["#667eea","#f093fb","#4facfe"],
            text=[f"{p*100:.1f}%" for p in probs],
            textposition="outside"
        ))
        fig.update_layout(
            yaxis_title="Probability (%)", xaxis_title="Species",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Adjust the sliders on the left and click **Predict** to see results.")
