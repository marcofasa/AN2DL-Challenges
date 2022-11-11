from tqdm import tqdm
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DatasetHelper:
    def __init__(self, path,seed):
        self.dataset_folder = path
        self.seed = seed

    #Convert ImageDataGenerator to Numpy
    def convert_dataset_to_numpy(self,dataset,dataset_size,batch_size):
        X = [] #Training
        Y = [] #Testing
        dataset_size = 3542
        for j in tqdm(range(0,int(dataset_size/batch_size))):
            images,labels = next(dataset)
            for i in range(images.shape[0]):
                X.append(images[i])
                Y.append(labels[i])

        X = np.array(X)
        Y = np.array(Y)

        return X,Y

    #Load Dataset Without Image augmentation
    def load_Dataset(self,image_size):
        train_data_gen = ImageDataGenerator()
        training_dir = self.dataset_folder
        batch_size = 8
        print("Extracting data from dataset at: " + training_dir)
        train_data = train_data_gen.flow_from_directory(directory=training_dir,
                                                        target_size=(96,96), #TODO change to image_size
                                                        color_mode='rgb',
                                                        classes=None, # can be set to labels
                                                        class_mode='categorical',
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=self.seed)
        #TODO SEE HOW RETRIVE DATASET SIZE FROM TRAIN_DATA
        return self.convert_dataset_to_numpy(train_data,3452,batch_size)

    #Load The dataset adding data throigh augmentation techniques
    def load_augmented_Dataset(self,image_size,augmentation_info): #TODO ADD THE AUGMENTATION PARAMETERS TO THIS FUNCTION
        train_data_gen = ImageDataGenerator()
        training_dir = self.path
        batch_size = 8
        train_data = train_data_gen.flow_from_directory(directory=training_dir,
                                                        target_size=(96,96), #TODO change to image_size
                                                        color_mode='rgb',
                                                        classes=None, # can be set to labels
                                                        class_mode='categorical',
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=self.seed)

        return self.convert_dataset_to_numpy(train_data,3452,batch_size)

    '''
        Normalize Dataset Data
        MODES:
        -1) Divide by max
        -2) Multiply by max
        -3) TODO ADD NEW METHODS FROM SLIDE
    '''
    def normalize_data(train,test,mode=1): #TODO PUT AN ENUMERATION FOR THE NORMALIZATION TYPE

        if mode==1:
            # Normalize data
            train = train/255. #pixel value
            test = test/255. #pixel value
        elif mode==2:
            train = train*255. #pixel value
            test = test*255. #pixel value
        #elif mode==3:

        return train,test