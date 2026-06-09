
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Load models (cached so they load only once) ──────────────
@st.cache_resource
def load_model():
    with open("models/iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/iris_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/iris_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, scaler, encoder

# ── Load dataset (cached) ─────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/IRIS.csv")
    return df

# ── Predict single flower ─────────────────────────────────────
def predict_flower(features):
    model, scaler, encoder = load_model()
    features_scaled = scaler.transform([features])
    prediction      = model.predict(features_scaled)
    probabilities   = model.predict_proba(features_scaled)[0]
    species         = encoder.inverse_transform(prediction)[0]
    confidence      = probabilities.max() * 100
    return species, confidence, probabilities, encoder.classes_

# ── Species metadata ──────────────────────────────────────────
SPECIES_INFO = {
    "Iris-setosa": {
        "emoji"      : "🌸",
        "color"      : "#FF6B9D",
        "description": "Small, compact flower found in Arctic and alpine regions.",
        "fact"       : "Easiest to identify — has the smallest petals!"
    },
    "Iris-versicolor": {
        "emoji"      : "🌼",
        "color"      : "#A78BFA",
        "description": "Medium-sized flower native to North America.",
        "fact"       : "Also called the 'Blue Flag' iris — very common in wetlands."
    },
    "Iris-virginica": {
        "emoji"      : "🌺",
        "color"      : "#34D399",
        "description": "Large flower native to eastern North America.",
        "fact"       : "Has the largest petals of all three species!"
    }
}
