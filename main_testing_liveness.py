import cv2 as cv
import numpy as np
import os
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from collections import defaultdict
import pickle
from numpy.linalg import norm
import mediapipe as mp
from skimage import feature
import dlib

# ------------------------
# CONFIG
# ------------------------
EMBEDDING_FILE = "faces_embeddings_done_4classes (3).npz"
SVM_MODEL_FILE = "svm_model_160x160.pkl"
HAAR_FILE = "haarcascade_frontalface_default.xml"
THRESHOLD = 1.0
VIDEO_DEVICE = 0

# ------------------------
# LOAD MODELS
# ------------------------

facenet = FaceNet()
detector_mtcnn = MTCNN()
haarcascade = cv.CascadeClassifier(HAAR_FILE) if os.path.exists(HAAR_FILE) else None

use_svm = os.path.exists(SVM_MODEL_FILE)
model = pickle.load(open(SVM_MODEL_FILE, 'rb')) if use_svm else None

# embeddings database
data = np.load(EMBEDDING_FILE, allow_pickle=True)
all_embeddings = data['arr_0']
all_labels = data['arr_1']

known_embeddings = defaultdict(list)
for emb, lbl in zip(all_embeddings, all_labels):
    known_embeddings[lbl].append(emb)

# ------------------------
# ANTI-SPOOFING HEURISTICS
# ------------------------

# 1. dlib for blink detection
detector_dlib = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

EAR_THRESHOLD = 0.21
BLINK_CONSEC_FRAMES = 2

blink_counter = 0
blink_detected = False

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# 2. Texture Liveness (LBP)
def analyze_texture(face_roi):
    gray = cv.cvtColor(face_roi, cv.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 58), range=(0, 58))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-5)
    return np.sum(hist[:10]) > 0.3   # True = real face

# 3. MediaPipe Hand Gesture
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
def detect_hand(frame):
    res = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    return res.multi_hand_landmarks is not None

# ------------------------
# HELPER – OPEN SET
# ------------------------
def is_unknown(embedding, known_embeddings, threshold=THRESHOLD):
    min_dist = float("inf")
    best_label = None

    for label, emb_list in known_embeddings.items():
        arr = np.stack(emb_list)
        d = np.linalg.norm(arr - embedding, axis=1).min()
        if d < min_dist:
            min_dist = d
            best_label = label

    if min_dist > threshold:
        return "Unknown", min_dist
    return best_label, min_dist

# ------------------------
# MAIN LOOP
# ------------------------

cap = cv.VideoCapture(VIDEO_DEVICE)

while True:
    ret, frame = cap.read()
    if not ret: break

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect face
    results = detector_mtcnn.detect_faces(rgb)
    if len(results) == 0 and haarcascade is not None:
        faces = haarcascade.detectMultiScale(gray, 1.3, 5)
        results = [{'box': (x, y, w, h)} for (x, y, w, h) in faces]

    # ------------- DLIB BLINK DETECTION -------------
    faces_dlib = detector_dlib(gray)
    for f in faces_dlib:
        shape = predictor(gray, f)
        leftEye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
        rightEye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])

        ear_left = calculate_ear(leftEye)
        ear_right = calculate_ear(rightEye)
        ear = (ear_left + ear_right) / 2

        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_CONSEC_FRAMES:
                blink_detected = True
            blink_counter = 0

    # ------------- RECOGNITION -------------
    for res in results:
        x, y, w, h = res['box']
        x, y = max(0, x), max(0, y)
        face_crop = frame[y:y+h, x:x+w]

        # Heuristic 1: blink
        if not blink_detected:
            cv.putText(frame, "NO BLINK", (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            continue

        # Heuristic 2: LBP texture
        if not analyze_texture(face_crop):
            cv.putText(frame, "FAKE TEXTURE", (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            continue

        # Heuristic 3: Hand gesture
        if not detect_hand(frame):
            cv.putText(frame, "SHOW HAND", (x, y-10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            continue

        # If all heuristics passed → do FaceNet
        face_rgb = cv.cvtColor(face_crop, cv.COLOR_BGR2RGB)
        face_resized = cv.resize(face_rgb, (160, 160))
        emb = facenet.embeddings(np.expand_dims(face_resized.astype('float32'), axis=0))[0]

        label, dist = is_unknown(emb, known_embeddings)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv.putText(frame, f"{label} [{dist:.2f}]", (x, y-10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv.imshow("Scanner", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
