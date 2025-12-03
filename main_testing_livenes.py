import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pickle
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from numpy.linalg import norm
from collections import defaultdict

# -------------------------
# CONFIG
# -------------------------
EMBEDDING_FILE = "faces_embeddings_done_4classes_with_live_model.npz"  # arr_0 (embeddings), arr_1 (labels)
SVM_MODEL_FILE  = "svm_model_160x160.pkl"      # opsional
HAAR_FILE       = "haarcascade_frontalface_default.xml"
LIVENESS_MODEL_FILE = "model.h5"               # ⬅️ taruh model liveness di folder kerja VS Code
THRESHOLD = 1.0
VIDEO_DEVICE = 0
# -------------------------

# --- Load liveness model lokal ---
if not os.path.exists(LIVENESS_MODEL_FILE):
    raise FileNotFoundError(f"Liveness model tidak ditemukan: {LIVENESS_MODEL_FILE}")
liveness_model = tf.keras.models.load_model(LIVENESS_MODEL_FILE)

# helper liveness (ikutin input size model)
def check_liveness(face_rgb: np.ndarray, model=liveness_model, thr=0.5) -> bool:
    """
    face_rgb: crop wajah format RGB (H,W,3)
    return True jika live, False jika spoof
    """
    # pastikan float [0..1]
    img = face_rgb.astype("float32") / 255.0
    # sesuaikan ke input model (H,W) dari model summary
    in_h, in_w = model.input_shape[1], model.input_shape[2]
    img = cv.resize(img, (in_w, in_h))
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)[0][0]
    return pred > thr

# Inisialisasi FaceNet & detektor
facenet = FaceNet()
detector_mtcnn = MTCNN()
haarcascade = cv.CascadeClassifier(HAAR_FILE) if os.path.exists(HAAR_FILE) else None

# (opsional) load SVM
use_svm = os.path.exists(SVM_MODEL_FILE)
model_svm = pickle.load(open(SVM_MODEL_FILE, 'rb')) if use_svm else None

# Load embeddings & labels training
data = np.load(EMBEDDING_FILE, allow_pickle=True)
all_embeddings = data['arr_0']
all_labels = data['arr_1']

# Build dict label -> list of embeddings
known_embeddings = defaultdict(list)
for emb, lbl in zip(all_embeddings, all_labels):
    known_embeddings[lbl].append(emb)

def is_unknown(embedding, known_embeddings, threshold=THRESHOLD):
    min_dist = float("inf")
    best_label = None
    for label, emb_list in known_embeddings.items():
        arr = np.stack(emb_list)            # (k,D)
        dists = np.linalg.norm(arr - embedding, axis=1)
        local_min = dists.min()
        if local_min < min_dist:
            min_dist = local_min
            best_label = label
    if min_dist > threshold:
        return "Unknown", min_dist
    else:
        return best_label, min_dist

# Kamera loop
cap = cv.VideoCapture(VIDEO_DEVICE)
if not cap.isOpened():
    raise RuntimeError("Tidak bisa membuka kamera. Cek device index / permission.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb  = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    results = detector_mtcnn.detect_faces(rgb)

    # fallback haarcascade
    if len(results) == 0 and haarcascade is not None:
        faces = haarcascade.detectMultiScale(gray, 1.3, 5)
        results = [{'box': (x, y, w, h)} for (x, y, w, h) in faces]

    for res in results:
        x, y, w, h = res['box']
        x, y = max(0, x), max(0, y)
        face_crop = rgb[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue

        # 1) Liveness check dulu
        is_live = check_liveness(face_crop)  # True = live, False = spoof
        if not is_live:
            # Merah: spoof
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv.putText(frame, "Spoofing detected!", (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv.LINE_AA)
            continue  # skip pengenalan

        # 2) FaceNet embedding (hanya untuk live)
        face_resized = cv.resize(face_crop, (160, 160))
        inp = face_resized.astype('float32')
        inp = np.expand_dims(inp, axis=0)
        embedding = facenet.embeddings(inp)[0]

        # 3) (opsional) SVM pred (untuk info tambahan)
        svm_name = None
        if use_svm:
            svm_pred = model_svm.predict(np.expand_dims(embedding, axis=0))[0]
            svm_name = svm_pred

        # 4) Open-set decision
        best_label, min_dist = is_unknown(embedding, known_embeddings, THRESHOLD)

        # Hijau: live & recognized/unknown
        label_text = best_label if svm_name is None else f"{best_label} (svm:{svm_name})"
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, f"{label_text} [{min_dist:.2f}]", (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv.LINE_AA)

    cv.imshow("Face Recognition + Liveness", frame)
    if cv.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
