import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


# 1. Load embeddings hasil FaceNet
data = np.load('faces_embeddings_done_4classes.npz')
X, y = data['arr_0'], data['arr_1']  # arr_0 = embeddings, arr_1 = label asli

# 2. Encode label menjadi angka
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 3. Latih SVM
model = SVC(kernel='linear', probability=True)
model.fit(X, y_encoded)

# 4. Simpan model SVM ke file .pkl
with open('svm_model_160x160.pkl', 'wb') as f:
    pickle.dump(model, f)

print("SVM model saved to svm_model_160x160.pkl")
