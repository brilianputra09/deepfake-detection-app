import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

def preprocess_efficientnet(image, img_size=(224, 224)):
    # Resize sesuai img_size
    img_resized = cv2.resize(image, img_size)

    # Pastikan dtype float32
    img_resized = img_resized.astype(np.float32)

    # Sesuaikan dengan preprocessing EfficientNet
    img_preprocessed = preprocess_input(img_resized)

    # Tambahkan batch dimension
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

    return img_preprocessed
