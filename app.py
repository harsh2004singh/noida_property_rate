import streamlit as st
import pickle
import traceback
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Noida House Price Predictor", page_icon="🏠")

st.title("🏠 Noida House Price Predictor")
st.write("Enter property details to estimate price (in Lakhs)")

# ---------------- AREA LIST ---------------- #
area = [
'extension, noida, noida','other','sector 100, noida','sector 101, noida',
'sector 102, noida','sector 104, noida','sector 107, noida','sector 108, noida',
'sector 110, noida','sector 117, noida','sector 119, noida','sector 120, noida',
'sector 121, noida','sector 122, noida','sector 128, noida','sector 129, noida',
'sector 130, noida','sector 131, noida','sector 134, noida','sector 135, noida',
'sector 137, noida','sector 143, noida','sector 143b, noida','sector 144, noida',
'sector 146, noida','sector 150, noida','sector 151, noida','sector 152, noida',
'sector 168, noida','sector 16b, noida','sector 25 yamuna express way, noida',
'sector 32, noida','sector 34, noida','sector 37, noida','sector 43, noida',
'sector 44, noida','sector 45, noida','sector 46, noida','sector 49, noida',
'sector 50, noida','sector 51, noida','sector 52, noida','sector 53, noida',
'sector 61, noida','sector 62, noida','sector 70, noida','sector 72, noida',
'sector 73, noida','sector 74, noida','sector 75, noida','sector 76, noida',
'sector 77, noida','sector 78, noida','sector 79, noida','sector 82, noida',
'sector 86, noida','sector 89, noida','sector 93, noida','sector 93a, noida',
'sector 93b, noida','sector 94, noida','yamuna expressway, noida, noida'
]

# ---------------- LOAD MODEL SAFELY ---------------- #
import urllib.request

MODEL_URL = "PASTE_YOUR_RELEASE_LINK_HERE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Ridge.pkl")

# Download model if not present
if not os.path.exists(MODEL_PATH):
    st.warning("Downloading model... please wait ⏳")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded!")

# Load model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# ---------------- INPUT FIELDS ---------------- #
st.header("Enter Property Details")

selected_area = st.selectbox("Select Area", sorted(area))
sqft = st.number_input("Square Feet", min_value=200, max_value=20000, step=50)
bhk = st.number_input("BHK", min_value=1, max_value=10, step=1)
bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)

# ---------------- PREDICT ---------------- #
if st.button("Predict Price 💰"):
    try:
        input_df = pd.DataFrame(
            [[sqft, bathroom, bhk, selected_area]],
            columns=['size', 'bathrooms', 'bedroom', 'address']
        )

        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)

        st.success(f"🏡 Estimated Price: ₹ {prediction} Lakh")

    except Exception as e:
        st.error("Prediction failed")
        st.code(traceback.format_exc())

