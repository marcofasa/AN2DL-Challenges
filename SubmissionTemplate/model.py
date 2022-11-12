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
        #TODO ADD NORMALIZATION OF X!!!!!
        X = self.preprocessing(X)
        out = self.model.predict(X)
        out = tf.argmax(out, axis=-1)

        return out

    #Preprocess data before apply prediction
    def preprocessing(self,X):
        X = X/255
        return X