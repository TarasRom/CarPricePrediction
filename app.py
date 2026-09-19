import joblib
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model


# Load model and preprocessor
@st.cache_resource
def load_model_and_resources():
    model = load_model("car_price_model.keras", compile=False)
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model_and_resources()

# User Interface
st.title("🚗 Car Price Prediction")

brand = st.selectbox(
    "Car brand",
    ["Toyota", "BMW", "Ford", "Audi", "Hyundai", "Mercedes", "Nissan", "Volkswagen", "Kia", "Honda"]
)

transmission = st.radio(
    "Transmission type",
    ["Manual", "Automatic", "Semi-Automatic"]
)

age = st.number_input(
    "Car age (years)",
    min_value=0,
    max_value=30,
    value=5,
    step=1
)

engine_size = st.number_input(
    "Engine size (liters)",
    min_value=1.0,
    max_value=5.0,
    value=2.0,
    step=0.1
)

mileage = st.slider(
    "Mileage (km)",
    min_value=1000,
    max_value=300000,
    value=50000,
    step=1000
)

# Preprocessing
def preprocess_input(brand, transmission, age, engine_size, mileage, preprocessor):
    df = pd.DataFrame({
        "Brand": [brand],
        "Transmission": [transmission],
        "Engine_Size": [engine_size],
        "Mileage": [mileage],
        "Age": [age]
    })

    df_processed = preprocessor.transform(df)

    return df_processed

# Prediction
if st.button("Predict Price"):
    input_data = preprocess_input(brand, transmission, age, engine_size, mileage, preprocessor)
    pred = model.predict(input_data, verbose=0)
    st.success(f"Estimated car price: ${pred[0][0]:,.2f}")
