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

# Tambahan Hans Liveness
import liveness as lvns

# -------------------------
# CONFIG
# -------------------------
EMBEDDING_FILE = "faces_embeddings_done_4classes.npz"  # berisi arr_0 (embeddings), arr_1 (labels)
SVM_MODEL_FILE = "svm_model_160x160.pkl"  # kalau pake SVM (opsional)
HAAR_FILE = "haarcascade_frontalface_default.xml"
THRESHOLD = 1.0   # threshold euclidean untuk "Unknown" (coba 0.9 - 1.2)
VIDEO_DEVICE = 0  # ganti 1 untuk kamera eksternal
# -------------------------

# Inisialisasi FaceNet & detektor
facenet = FaceNet()
detector_mtcnn = MTCNN()
haarcascade = cv.CascadeClassifier(HAAR_FILE) if os.path.exists(HAAR_FILE) else None

# (Opsional) load SVM jika ingin kombinasi classifier
use_svm = os.path.exists(SVM_MODEL_FILE)
model = pickle.load(open(SVM_MODEL_FILE, 'rb')) if use_svm else None

# Load embeddings training & labels dari .npz
data = np.load(EMBEDDING_FILE, allow_pickle=True)
all_embeddings = data['arr_0']   # NxD (pastikan ini benar)
all_labels = data['arr_1']       # N labels (string)

# Build dict {label: [emb1, emb2, ...]}
known_embeddings = defaultdict(list)
for emb, lbl in zip(all_embeddings, all_labels):
    known_embeddings[lbl].append(emb)

# Helper: fungsi prediksi unknown berdasarkan dict per-label
def is_unknown(embedding, known_embeddings, threshold=THRESHOLD):
    """
    embedding : 1D numpy array (D,)
    known_embeddings : dict {label: [emb1, emb2, ...]}
    return: (best_label_or_Unknown, min_distance)
    """
    min_dist = float("inf")
    best_label = None
    for label, emb_list in known_embeddings.items():
        # vectorized compute distances for this label (faster than Python loop for many emb)
        arr = np.stack(emb_list)            # shape (k, D)
        # compute euclidean distances to all embeddings for this label
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
    raise RuntimeError("Tidak bisa membuka kamera. Cek device index.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect with MTCNN first
    results = detector_mtcnn.detect_faces(rgb)

    # fallback ke haarcascade bila MTCNN tidak menemukan
    if len(results) == 0 and haarcascade is not None:
        faces = haarcascade.detectMultiScale(gray, 1.3, 5)
        results = [{'box': (x, y, w, h)} for (x, y, w, h) in faces]

    for res in results:
        x, y, w, h = res['box']
        x, y = max(0, x), max(0, y)
        face_crop = rgb[y:y+h, x:x+w]

        print(face_crop.shape)
        if face_crop.size == 0:
            continue

        # TAMBAHAN HANS UNTUK CHECK LIVENESS
        # ---------------- LIVENESS CHECK ----------------
        bbox = [x, y, w, h]
        is_real = lvns.check_liveness(frame, bbox)

        if not is_real:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  
            cv.putText(
                frame,
                "SPOOF",
                (x, y-10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
                cv.LINE_AA
            )
            continue  

        # ------------------------------------------------

        # Preprocess face untuk FaceNet
        face_resized = cv.resize(face_crop, (150, 150))
        # FaceNet expects shape (1, H, W, C) and type float32; leave value scale as-is (keras-facenet handles)
        inp = face_resized.astype('float32')
        inp = np.expand_dims(inp, axis=0)

        # get embedding
        embedding = facenet.embeddings(inp)[0]  # 1D array

        # optional: get SVM pred (not used for open-set decision, but can show)
        svm_name = None
        if use_svm:
            svm_pred = model.predict(np.expand_dims(embedding, axis=0))[0]
            # svm_pred may be encoded int - if you saved encoder use it to map back
            svm_name = svm_pred

        # open-set decision using embedding distances
        best_label, min_dist = is_unknown(embedding, known_embeddings, THRESHOLD)
        display_name = best_label
        # optional: append svm result for debugging
        if svm_name is not None:
            display_name = f"{display_name} (svm:{svm_name})"

        # draw box + label
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv.putText(frame, f"{display_name} [{min_dist:.2f}]", (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)

    cv.imshow("Face Recognition scanning", frame)
    if cv.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

cap.release()
cv.destroyAllWindows()
