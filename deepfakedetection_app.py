import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

from utils.mediapipe_eye_crop import crop_eyes_mediapipe
from utils.preprocess import preprocess_efficientnet

# =====================
# KONFIGURASI MODEL
# =====================
FILE_ID = "1Sz_11M06ztbozdzqhW4JovYS19KkZX8P" 
MODEL_PATH = "Final_EfficientNetB0_Model.h5"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive...")
        gdown.download(URL, MODEL_PATH, quiet=False)

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()
CLASS_NAMES = ["Fake", "Real"]

# =====================
# UI
# =====================
st.set_page_config(page_title="Real vs Fake Eye Detection", layout="centered")
st.title("🔍 Deteksi Gambar Real / Fake")
st.write("Berdasarkan **area mata** menggunakan EfficientNet + MediaPipe")

uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Gambar Asli", use_column_width=True)

    # Crop mata
    eye_img = crop_eyes_mediapipe(image_np)

    if eye_img is None:
        st.error("❌ Area mata tidak terdeteksi")
    else:
        st.image(eye_img, caption="Area Mata (Cropped)", width=300)

        # Preprocess untuk EfficientNet
        input_tensor = preprocess_efficientnet(eye_img)

        # Predict
        preds = model.predict(input_tensor, verbose=0)
        prob = float(preds[0][0])
        label = "Real" if prob > 0.5 else "Fake"

        st.write(f"**Kelas:** {label}")
        st.write(f"**Probabilitas:** {prob:.2%}")
