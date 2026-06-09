
import streamlit as st

st.set_page_config(page_title="About", page_icon="👨‍💻", layout="wide")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("👨‍💻 About Developer")

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("""
    <div class="glass-card" style="text-align:center;">
        <h1>👩‍💻</h1>
        <h2>Akshaya Vasireddi</h2>
        <p>Machine Learning Engineer</p>
        <span class="badge badge-success">Available for Opportunities</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <h4>🚀 About This Project</h4>
        <p>This is a production-quality ML web application built to demonstrate
        end-to-end machine learning engineering skills — from data exploration
        and model training to deployment and explainability.</p>
        <br>
        <h4>🛠️ Tech Stack</h4>
        <p>
            <span class="badge badge-info">Python</span>&nbsp;
            <span class="badge badge-info">Streamlit</span>&nbsp;
            <span class="badge badge-info">Scikit-learn</span>&nbsp;
            <span class="badge badge-info">SHAP</span>&nbsp;
            <span class="badge badge-info">Plotly</span>&nbsp;
            <span class="badge badge-info">Pandas</span>&nbsp;
            <span class="badge badge-info">XGBoost</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.subheader("🏆 Key Skills Demonstrated")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""<div class="glass-card">
        <h4>📊 Data Science</h4>
        <ul>
            <li>EDA & Visualization</li>
            <li>Feature Engineering</li>
            <li>Statistical Analysis</li>
        </ul>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="glass-card">
        <h4>🤖 Machine Learning</h4>
        <ul>
            <li>5 ML Algorithms</li>
            <li>Model Comparison</li>
            <li>Hyperparameter Tuning</li>
        </ul>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class="glass-card">
        <h4>🚀 MLOps & Deployment</h4>
        <ul>
            <li>Model Serialization</li>
            <li>Streamlit Cloud Deploy</li>
            <li>Explainable AI (SHAP)</li>
        </ul>
    </div>""", unsafe_allow_html=True)
