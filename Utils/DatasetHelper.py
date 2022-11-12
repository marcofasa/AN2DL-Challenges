from tqdm import tqdm
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
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

        x=np.concatenate([dataset.next()[0] for i in tqdm(range(dataset_size))])
        y=np.concatenate([dataset.next()[1] for i in tqdm(range(dataset_size))])
        '''
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
        '''
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

    '''
        Return Xtrain,X_val,X_test,Ytrain,Y_test,Y_val

        specify the split for test and validation and specify the normalization mode 
        (see self.normalize_data for the modality available)
    '''
    def split_and_normalize(self,X,Y,split_test = .1 ,split_val =.1,normalization_mode=1):
        #Split Training and Testing
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=self.seed, test_size=int(split_test * X.shape[0]),stratify = Y)

        # Normalize data
        X_train,X_test = self.normalize_data(X_train,X_test,1)

        #Split Training and Validation
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, random_state=self.seed, test_size=int(split_val * X_train.shape[0]),stratify = Y_train)


        return X_train,X_test,X_val,Y_train,Y_test,Y_val

    #Generate a new X,Y with augmented data of "num_of_images"
    #TODO ADD SOME PARAMETER TO CHANGE AUGMENTATION TYPE
    def apply_data_augmentation(self,X,Y,num_of_images,norm_mode = 1):
        X = self.denormalize(X,norm_mode) #Denormalize
        print("BB")
        #TODO PARAMETRIZE THIS PART
        data_generator = ImageDataGenerator(
            rotation_range = 15,
            shear_range = 0.2,
            zoom_range = 0.3,
            brightness_range = (0.5, 1.5)
            )
        
        i=0
        batch_size = 32
        stop_condition =  int(num_of_images / batch_size)
        print("STOP CONDITION; " + str(stop_condition))

        generator = data_generator.flow(
                                    X,
                                    Y,
                                    batch_size=32,
                                    shuffle=True,
                                    sample_weight=None,
                                    seed=self.seed,
                                    save_to_dir=None,
                                    save_format='png',
                                    ignore_class_split=False,
                                    subset=None
                                )

        print(X.shape)
        generator.reset()
        for i in tqdm(range(stop_condition)):
            imgages,targets = generator.next()
            X=np.concatenate((X,imgages), axis=0)
            Y=np.concatenate((Y,targets), axis=0)


        #x=np.concatenate([generator.next()[0] for i in tqdm(range(stop_condition))])
        #y=np.concatenate([generator.next()[1] for i in tqdm(range(stop_condition))])


        #X = np.concatenate((X, x), axis=0)
        #Y = np.concatenate((Y, y), axis=0)

        print(X.shape)
        print(Y.shape)
        
        X = self.normalize(X,norm_mode)
        return X,Y
        

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
    def normalize_data(self,train,test,mode=1): #TODO PUT AN ENUMERATION FOR THE NORMALIZATION TYPE

        train = self.normalize(train,mode)
        test  = self.normalize(test,mode)

        #TODO image mean normalization, image deviation normalization etc... see slide
        return train,test

    def denormalize(self,X,mode=1):
        if mode==1:
            # Normalize data
            X = X*255. #pixel value
        elif mode==2:
            X = X/255
        #elif mode==3:

        return X

    def normalize(self,X,mode=1):
        if mode==1:
            # Normalize data
            X = X/255. #pixel value
        elif mode==2:
            X = X*255
        #elif mode==3:

        return X
