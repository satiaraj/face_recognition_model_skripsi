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

EMBEDDING_FILE = "faces_embeddings_done_4classes (4).npz"
data = np.load(EMBEDDING_FILE, allow_pickle=True)
embeddings = data['arr_0']

print("Shape embeddings:", embeddings.shape)
