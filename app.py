import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model, scaler, columns
@st.cache_resource
def load_model_and_resources():
    model = load_model("car_price_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

model, scaler, columns = load_model_and_resources()


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

# Alternative
# transmission = st.selectbox("Transmission type", ["Manual", "Automatic", "Semi-Automatic"])

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
def preprocess_input(brand, transmission, age, engine_size, mileage, scaler, columns):
    df = pd.DataFrame({
        "Brand": [brand],
        "Transmission": [transmission],
        "Age": [age],
        "Engine_Size": [engine_size],
        "Mileage": [mileage]
    })

    # One-hot encoding of categorical variables
    df = pd.get_dummies(df, columns=['Brand', 'Transmission'], dtype=int)

    # Add missing columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training data
    df = df[columns]

    # Normalize numeric features (using the same scaler from training)
    numeric_cols = ['Engine_Size', 'Mileage', 'Age']
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df

#  Prediction
if st.button("Predict Price"):
    input_data = preprocess_input(brand, transmission, age, engine_size, mileage, scaler, columns)
    pred = model.predict(input_data)
    st.success(f"Estimated car price: ${pred[0][0]:,.2f}")
