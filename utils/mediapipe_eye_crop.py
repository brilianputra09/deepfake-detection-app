import cv2
import mediapipe as mp

# =========================
# Inisialisasi MediaPipe
# =========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.9
)

# =========================
# Landmark Mata
# =========================
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
            133, 173, 157, 158, 159, 160, 161, 246]

RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
             263, 466, 388, 387, 386, 385, 384, 398]

ALL_EYES_LANDMARKS = LEFT_EYE + RIGHT_EYE

# =========================
# Fungsi Cropping
# =========================
def crop_eyes_mediapipe(image):
    if image is None:
        return None

    h, w, _ = image.shape

    # MediaPipe butuh RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    face_landmarks = results.multi_face_landmarks[0]

    # Ambil semua landmark mata
    eye_points = [
        (
            int(face_landmarks.landmark[i].x * w),
            int(face_landmarks.landmark[i].y * h)
        )
        for i in ALL_EYES_LANDMARKS
    ]

    x_coords = [p[0] for p in eye_points]
    y_coords = [p[1] for p in eye_points]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Padding
    padding = 25
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    cropped_eye_region = image[y_min:y_max, x_min:x_max]

    if cropped_eye_region.size == 0:
        return None

    return cropped_eye_region
