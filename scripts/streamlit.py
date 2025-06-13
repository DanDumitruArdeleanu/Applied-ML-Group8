import streamlit as st
import requests
import os
import pyautogui
from io import BytesIO
from PIL import Image

st.title("""Predicting Surface Normal Vectors""")
st.write("""Group 8 - Dan Ardeleanu, Adna Kapidzic, Özde Pilli, Dorin-Vlad Udrea""")

github_url = "https://github.com/DanDumitruArdeleanu/Applied-ML-Group8"
st.markdown(f"[Github Repository]({github_url}) - Run the FastAPI using the Readme file description")

API_url = "http://127.0.0.1:8000/docs"
st.markdown(f"[FastAPI]({API_url}) - View the FastAPI")

predict_url = "http://localhost:8000/predict/"

uploaded_file = st.file_uploader(
    "Upload PNG of an Object",
    type=["png"],
    help="Only .png files are allowed"
)

if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded PNG", use_container_width=True)

if st.button("Predict", type="primary"):
    if uploaded_file is not None:
        "Predicted Normal of the Object"
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/png")}
        resp = requests.post(predict_url, files=files, timeout=10)
        resp.raise_for_status()

        pred_img = Image.open(BytesIO(resp.content))
        st.image(pred_img, caption="Predicted Normal Map", use_container_width=True)

        buffer = BytesIO()
        pred_img.save(buffer, format="PNG")
        buffer.seek(0)

        st.download_button(
            label="Save Prediction",
            data=buffer,
            file_name=f"{uploaded_file.name.split('.')[0]}_normal_map.png",
            mime="image/png",
            key="download_pred"
)

if st.button("Reset"):
     pyautogui.hotkey("ctrl","F5")