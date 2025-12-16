import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import requests

from utils.mediapipe_eye_crop import crop_eyes_mediapipe
from utils.preprocess import preprocess_efficientnet

# =====================
# KONFIGURASI MODEL
# =====================
MODEL_PATH = "Final_EfficientNetB0_Model.h5"
HF_MODEL_URL = "https://huggingface.co/brilianputra09/Deepfake-Detection-EfficientNet-Model/blob/main/Final_EfficientNetB0_Model.h5"

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Mengunduh model dari Hugging Face...")
        try:
            r = requests.get(HF_MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("✅ Model berhasil diunduh")
        except Exception as e:
            st.error(f"❌ Gagal mengunduh model: {e}")
            return None

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
