import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt
# Judul Aplikasi
st.title("Analisis Model SVM Face Recognition")

# 1. Load embeddings hasil FaceNet
uploaded_file = st.file_uploader("Upload file embeddings (.npz)", type=["npz"])
if uploaded_file is not None:
    data = np.load(uploaded_file)
    X, y = data['arr_0'], data['arr_1']

    # 2. Encode label
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # 3. Parameter Grid Search
    st.subheader("🔍 Grid Search untuk Hyperparameter")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5)
    grid.fit(X, y_encoded)

    # Menampilkan semua kombinasi parameter yang diuji GridSearch
    st.subheader("📋 Hasil Lengkap Grid Search")
    results_df = pd.DataFrame(grid.cv_results_)
    cols = [
        'param_C', 'param_gamma', 'param_kernel',
        'split0_test_score', 'split1_test_score', 'split2_test_score',
        'split3_test_score', 'split4_test_score',
        'mean_test_score', 'std_test_score'
    ]
    results_df = results_df[cols]

    # Urutkan dari kombinasi dengan rata-rata tertinggi
    results_df = results_df.sort_values(by='mean_test_score', ascending=True)

    st.dataframe(results_df)
    

    st.write("**Best Parameters:**", grid.best_params_)

    # 4. K-Fold Cross Validation
    st.subheader("📊 Hasil K-Fold Cross Validation")
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []
    fold_num = 1

    for train_idx, test_idx in kfold.split(X, y_encoded):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        model = SVC(**grid.best_params_, probability=True)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)
        
        st.markdown(f"### Fold {fold_num}")
        st.write(f"Akurasi: **{acc:.4f}**")
        
        # Classification report
        report = classification_report(y_test, y_pred, labels=np.arange(len(encoder.classes_)),target_names=encoder.classes_, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        # Visualisasi Precision, Recall, F1-score, Support
        metrics_df = report_df.iloc[:-3, :]  # ambil hanya kelas, bukan avg/total
        fig, ax = plt.subplots(figsize=(10,5))
        metrics_df[['precision','recall','f1-score']].plot(kind='bar', ax=ax)
        ax.set_title(f"Precision, Recall, F1-Score per Kelas - Fold {fold_num}")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.1)
        st.pyplot(fig)

        # Confusion Matrix dengan label nama
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(encoder.classes_)))
        cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
        st.write("Confusion Matrix (dengan nama):")
        st.dataframe(cm_df)
        
        fold_num += 1

    st.write("**Rata-rata Akurasi:**", np.mean(acc_scores))

    # 5. Latih ulang SVM di seluruh dataset untuk model final
    final_model = SVC(**grid.best_params_, probability=True)
    final_model.fit(X, y_encoded)

    # 6. Simpan model
    model_filename = 'svm_model_160x160.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(final_model, f)

    st.success(f"SVM model saved to {model_filename}")

# # 1. Load embeddings hasil FaceNet
# data = np.load('faces_embeddings_done_4classes_second_tes.npz')
# X, y = data['arr_0'], data['arr_1']  # arr_0 = embeddings, arr_1 = label asli

# # 2. Encode label menjadi angka
# encoder = LabelEncoder()
# y_encoded = encoder.fit_transform(y)

# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'linear']
# }
# grid = GridSearchCV(SVC(probability=True), param_grid, cv=5)
# grid.fit(X, y_encoded)
# print("Best Parameters:", grid.best_params_)

# # 3. K-Fold Cross Validation
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# acc_scores = []
# fold_num = 1

# for train_idx, test_idx in kfold.split(X, y_encoded):
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
#     # model = SVC(kernel='linear', probability=True)
#     model = SVC(**grid.best_params_, probability=True)
#     model.fit(X_train, y_train)
    
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     acc_scores.append(acc)
    
#     print(f"=== Fold {fold_num} ===")
#     print("Akurasi:", acc)
#     print(classification_report(y_test, y_pred, target_names=encoder.classes_))
#     cm = confusion_matrix(y_test, y_pred)
#     cm_df = pd.DataFrame(cm, index=encoder.classes_, columns=encoder.classes_)
#     print("Confusion Matrix (dengan nama):\n", cm_df)
#     print("\n")
#     # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#     # print("\n")
    
#     fold_num += 1

# print("Rata-rata Akurasi:", np.mean(acc_scores))

# # 4. Latih ulang SVM di seluruh dataset untuk model final
# # final_model = SVC(kernel='linear', probability=True)
# final_model = SVC(**grid.best_params_, probability=True)
# final_model.fit(X, y_encoded)

# # 5. Simpan model
# with open('svm_model_160x160.pkl', 'wb') as f:
#     pickle.dump(final_model, f)

# print("SVM model saved to svm_model_160x160(1).pkl")

# # 1. Load embeddings hasil FaceNet
# data = np.load('faces_embeddings_done_4classes.npz')
# X, y = data['arr_0'], data['arr_1']  # arr_0 = embeddings, arr_1 = label asli

# # 2. Encode label menjadi angka
# encoder = LabelEncoder()
# y_encoded = encoder.fit_transform(y)

# # 3. Latih SVM
# model = SVC(kernel='linear', probability=True)
# model.fit(X, y_encoded)

# # 4. Simpan model SVM ke file .pkl
# with open('svm_model_160x160.pkl', 'wb') as f:
#     pickle.dump(model, f)

# print("SVM model saved to svm_model_160x160.pkl")