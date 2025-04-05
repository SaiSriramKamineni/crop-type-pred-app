import streamlit as st
import joblib
import numpy as np

st.title("🌾 Crop Type Classification")

model = joblib.load("crop_type_model.pkl")
scaler = joblib.load("crop_type_scaler.pkl")
encoder = joblib.load("crop_type_encoder.pkl")

soil_pH = st.number_input("Soil pH", value=6.5)
rainfall = st.number_input("Rainfall (mm)", value=100.0)
temperature = st.number_input("Temperature (°C)", value=28.0)

if st.button("Predict Crop Type"):
    features = np.array([[soil_pH, rainfall, temperature]])
    prediction = model.predict(scaler.transform(features))
    crop = encoder.inverse_transform([int(round(prediction[0]))])[0]
    st.success(f"🌱 Predicted Crop Type: {crop}")
