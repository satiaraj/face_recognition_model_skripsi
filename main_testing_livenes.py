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

# Liveness
import liveness_check as lvns

# -------------------------
# CONFIG
# -------------------------
EMBEDDING_FILE = "faces_embeddings_done_4classes (3).npz"
SVM_MODEL_FILE = "svm_model_160x160.pkl"
THRESHOLD = 1.0
VIDEO_DEVICE = 0
# -------------------------

facenet = FaceNet()
detector_mtcnn = MTCNN()

# Load embeddings
data = np.load(EMBEDDING_FILE, allow_pickle=True)
all_embeddings = data["arr_0"]
all_labels = data["arr_1"]

known_embeddings = defaultdict(list)
for emb, lbl in zip(all_embeddings, all_labels):
    known_embeddings[lbl].append(emb)

# Function: open-set recognition
def is_unknown(embedding, known_embeddings, threshold=THRESHOLD):
    min_dist = float("inf")
    best_label = None
    
    for label, emb_list in known_embeddings.items():
        arr = np.stack(emb_list)
        dists = np.linalg.norm(arr - embedding, axis=1)
        local_min = dists.min()
        
        if local_min < min_dist:
            min_dist = local_min
            best_label = label
    
    if min_dist > threshold:
        return "Unknown", min_dist
    return best_label, min_dist


# -------------------------
# Tracking-based Loop
# -------------------------
cap = cv.VideoCapture(VIDEO_DEVICE)

tracker = None
track_box = None  # (x, y, w, h)
frame_count = 0
live_status = "UNKNOWN"   # REAL / SPOOF / UNKNOWN

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    frame_count += 1

    # ============================================
    # 1. Jika BELUM ADA tracker → DETECT wajah
    # ============================================
    if tracker is None:
        results = detector_mtcnn.detect_faces(rgb)
        
        if len(results) > 0:
            x, y, w, h = results[0]['box']
            x, y = max(0, x), max(0, y)

            track_box = (x, y, w, h)

            tracker = cv.legacy.TrackerCSRT_create()
            tracker.init(frame, track_box)
        else:
            cv.imshow("Face Recognition scanning", frame)
            if cv.waitKey(1) & 0xFF == 27: 
                break
            continue

    # ============================================
    # 2. Update tracker
    # ============================================
    ok, box = tracker.update(frame)
    if not ok:
        tracker = None
        live_status = "UNKNOWN"
        continue

    x, y, w, h = [int(v) for v in box]
    face_crop = rgb[y:y+h, x:x+w]

    if face_crop.size == 0:
        continue

    # ============================================
    # 3. Liveness check tiap 5 frame
    # ============================================
    if frame_count % 5 == 0:
        prob, label = lvns.check_liveness(face_crop)
        live_status = "REAL" if label == 1 else "SPOOF"

    # Tampilkan status liveness
    if live_status == "SPOOF":
        cv.putText(frame, "SPOOF", (x, y - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv.imshow("Face Recognition scanning", frame)
        if cv.waitKey(1) & 0xFF == 27: break
        continue

    cv.putText(frame, f"REAL", (x, y - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # ============================================
    # 4. Kalau REAL → lanjut FaceNet Recognition
    # ============================================
    face_resized = cv.resize(face_crop, (160, 160))
    inp = np.expand_dims(face_resized.astype("float32"), axis=0)

    embedding = facenet.embeddings(inp)[0]

    best_label, min_dist = is_unknown(embedding, known_embeddings)

    cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    cv.putText(frame, f"{best_label} [{min_dist:.2f}]", (x, y+h+20),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Show
    cv.imshow("Face Recognition scanning", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
