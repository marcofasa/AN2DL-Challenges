from tqdm import tqdm
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DatasetHelper:
    #Path is the folder where we will have dataset,model_savings....
    def __init__(self, path,seed):
        self.path = path
        self.dataset_folder = os.path.join(self.path, 'data')
        self.numpy_dataset  = os.path.join(self.path, 'data_numpy_format')
        self.local_checkpoints = os.path.join(self.path, 'local_checkpoints')
        self.local_tensorboard = os.path.join(self.path, 'local_tensorboard')
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


    #TODO ADD SOME PARAMETERS TO CHANGE
    def apply_data_augmentation(self,X,Y):
        X_new = []
        Y_new = []
        #TODO PARAMETRIZE THIS PART
        data_generator = ImageDataGenerator(
            rotation_range = 40,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            brightness_range = (0.5, 1.5)
            )
        
        i=0
        batch_size = 32
        stop_condition =  int(1000 / batch_size)
        print("STOP CONDITION; " + str(stop_condition))

        generator = data_generator.flow(
                                    X,
                                    Y,
                                    batch_size=32,
                                    shuffle=True,
                                    sample_weight=None,
                                    seed=self.seed,
                                    save_to_dir='./test_augmentation',
                                    save_format='png',
                                    ignore_class_split=False,
                                    subset=None
                                )

        print(X.shape)

        x=np.concatenate([generator.next()[0] for i in tqdm(range(stop_condition))])
        y=np.concatenate([generator.next()[1] for i in tqdm(range(stop_condition))])

        print(x.shape)
        print(y.shape)

        X = np.concatenate((X, x), axis=0)
        Y = np.concatenate((Y, y), axis=0)

        print(X.shape)
        print(Y.shape)
        

    #Allow to save all images directly in numpy format, no need to load them 1 by one (fasten up the data augmentation problem)
    #TODO COMPLETE THIS FUNCTION (load function dosnt work properly)
    def load_dataset_from_numpy(self):
            #Check if dataset in numpy format is present
            check_images_numpy  = os.path.isfile(os.path.join(self.numpy_dataset, "images.npy"))
            check_targets_numpy = os.path.isfile(os.path.join(self.numpy_dataset, "targets.npy"))


            if check_images_numpy and check_targets_numpy:
                #Load from numpy
                X = np.load(os.path.join(self.numpy_dataset, "images.npy"))
                Y = np.load(os.path.join(self.numpy_dataset, "targets.npy"))
                return X,Y
            else:
                #Load dataset using ImageDataGenerator
                X,Y = self.load_Dataset(10)
                
                #Save Numpy arrays to file
                np.save(os.path.join(self.numpy_dataset, 'images'), X)
                np.save(os.path.join(self.numpy_dataset, 'targets'),Y)
                #Return dataset
                return X,Y


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