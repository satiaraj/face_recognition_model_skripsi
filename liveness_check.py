import tensorflow as tf
# from tf_keras.models import load_model
# from tf_keras.applications.efficientnet import EfficientNetB0
# from efficientnet.tfkeras import preprocess_input
# from tensorflow.keras.layers import Layer
# from tf_keras.layers import Layer
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Layer


IMG_SIZE = (224, 224)

class CastLayer(Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

custom_objects = {"Cast": CastLayer}

model = load_model(
    "./Models/14_model.h5",
    custom_objects=custom_objects
)

def check_liveness(face):

    img = tf.image.resize(face, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img = tf.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    label = int(pred > 0.5)
    print(f"Pred: {pred}, Label: {label}")

    return label

