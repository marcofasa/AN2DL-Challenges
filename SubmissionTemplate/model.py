import tensorflow as tf
import numpy as np
import os
import random
from tensorflow import keras

class model:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(os.path.join(path, 'model1'))

    def predict(self, X):
        
        # Insert your preprocessing here

        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)

        return out