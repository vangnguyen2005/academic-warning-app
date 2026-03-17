import streamlit as st
import pickle
import pandas as pd

st.title("Academic Warning Prediction")

data = pickle.load(open("academic_model.pkl","rb"))
model = data["model"]
features = data["features"]
scaler = data["scaler"]

inputs = {}

for feature in features:
    inputs[feature] = st.number_input(feature, value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([inputs])
    df = df[features]

    df = scaler.transform(df)

    pred = model.predict(df)

    if pred[0] == 1:
        st.error("Nguy cơ cảnh báo")
    else:
        st.success("Bình thường")