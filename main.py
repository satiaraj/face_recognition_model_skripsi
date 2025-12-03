import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from numpy.linalg import norm

# INITIALIZE FACENET
facenet = FaceNet()

# Load training embeddings & labels
faces_embeddings = np.load("faces_embeddings_done_4classes_tes.npz")
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)

# Load SVM model
model = pickle.load(open("svm_model_160x160.pkl", 'rb'))

# Initialize detectors
# detector_mtcnn = MTCNN(min_face_size=20, steps_threshold=[0.5, 0.6, 0.6])  # lebih sensitif
detector_mtcnn = MTCNN()
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Kamera
cap = cv.VideoCapture(0)  # ganti 1 kalau kamera eksternal

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    results = detector_mtcnn.detect_faces(rgb_img)

    if len(results) == 0:
        faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)
        results = [{'box': (x, y, w, h)} for (x, y, w, h) in faces]

    for res in results:
        x, y, w, h = res['box']
        x, y = max(0, x), max(0, y)
        face_crop = rgb_img[y:y+h, x:x+w]

        if face_crop.size == 0:
            continue

        img = cv.resize(face_crop, (160, 160))
        img = np.expand_dims(img, axis=0)

        # === 🔹 Bagian modifikasi mulai di sini ===
        embedding = facenet.embeddings(img)

        pred_class = model.predict(embedding)[0]
        pred_name = encoder.inverse_transform([pred_class])[0]

# Simpan embeddings training dan labelnya
        known_embeddings = faces_embeddings['arr_0']  # Nx128
        known_labels = Y  # label string original

        # Saat prediksi
        distances = [norm(embedding[0] - e) for e in known_embeddings]
        min_dist = np.min(distances)
        best_label = known_labels[np.argmin(distances)]

        if min_dist > 1.0:  # threshold euclidean, coba antara 0.9 - 1.2
            pred_name = "Unknown"
        else:
            pred_name = best_label

        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv.putText(frame, str(pred_name), (x, y-10), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 0, 255), 2, cv.LINE_AA)

    cv.imshow("Face Recognition scanning", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()

