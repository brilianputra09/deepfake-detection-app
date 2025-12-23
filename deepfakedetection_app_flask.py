import os
import cv2
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Import custom utils (pastikan folder utils ada di direktori yang sama)
from utils.mediapipe_eye_crop import crop_eyes_mediapipe
from utils.preprocess import preprocess_efficientnet

app = Flask(__name__)

# Konfigurasi Upload Folder
UPLOAD_FOLDER = 'static/uploads'
CROP_FOLDER = 'static/crops'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =====================
# KONFIGURASI MODEL (LOGIKA TETAP SAMA)
# =====================
MODEL_PATH = "Final_EfficientNetB0_Model_RMSProp.h5"
HF_MODEL_URL = "https://huggingface.co/brilianputra09/Deepfake-Detection-EfficientNet-Model/resolve/main/Final_EfficientNetB0_Model_RMSProp.h5"

def load_trained_model():
    """Fungsi load model tanpa st.cache_resource"""
    if not os.path.exists(MODEL_PATH):
        print("Mengunduh model dari Hugging Face...")
        r = requests.get(HF_MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Model berhasil diunduh")
    
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# Load model di awal start server
print("Loading Model...")
model = load_trained_model()
print("Model Ready!")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Cek apakah ada file
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Proses Gambar (Sesuai Logika Lama)
            try:
                # 1. Buka Gambar
                image = Image.open(file_path).convert("RGB")
                image_np = np.array(image)

                # 2. Crop Mata
                eye_img = crop_eyes_mediapipe(image_np)

                if eye_img is None:
                    return render_template('index.html', 
                                           error="Area mata tidak terdeteksi. Gunakan gambar wajah yang jelas.", 
                                           original_image=file_path)

                # Simpan hasil crop untuk ditampilkan di HTML
                crop_filename = "crop_" + filename
                crop_path = os.path.join(CROP_FOLDER, crop_filename)
                
                # Convert BGR (OpenCV) ke RGB untuk Pillow atau sebaliknya jika perlu
                # eye_img dari mediapipe biasanya sudah RGB jika inputnya RGB
                Image.fromarray(eye_img).save(crop_path)

                # 3. Preprocess EfficientNet
                input_tensor = preprocess_efficientnet(eye_img)

                # 4. Predict
                preds = model.predict(input_tensor, verbose=0)
                prob = float(preds[0][0])
                label = "Real" if prob > 0.5 else "Fake"
                
                # Format probabilitas untuk UI
                prob_percent = f"{prob:.2%}"
                
                # Warna progress bar berdasarkan hasil
                result_color = "success" if label == "Real" else "danger"

                return render_template('index.html', 
                                       original_image=file_path,
                                       cropped_image=crop_path,
                                       prediction=label,
                                       probability=prob_percent,
                                       raw_prob=prob,
                                       result_color=result_color)

            except Exception as e:
                print(e)
                return render_template('index.html', error=f"Terjadi kesalahan saat memproses gambar: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
