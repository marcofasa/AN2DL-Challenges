from tqdm import tqdm
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import keras_cv as keras_cv


class DatasetHelper:
    # Path is the folder where we will have dataset, model_savings....
    def __init__(self, path, seed, create_dirs=False):
        self.path = path
        self.dataset_folder = os.path.join(self.path, 'data')
        self.numpy_dataset = os.path.join(self.path, 'data_numpy_format')
        self.local_checkpoints = os.path.join(self.path, 'local_checkpoints')
        self.local_tensorboard = os.path.join(self.path, 'local_tensorboard')

        if create_dirs:
            if not os.path.exists(self.dataset_folder):
                os.makedirs(self.dataset_folder)
            if not os.path.exists(self.numpy_dataset):
                os.makedirs(self.numpy_dataset)
            if not os.path.exists(self.local_checkpoints):
                os.makedirs(self.local_checkpoints)
            if not os.path.exists(self.local_tensorboard):
                os.makedirs(self.local_tensorboard)

        self.seed = seed

    # Convert ImageDataGenerator to Numpy
    def convert_dataset_to_numpy(self, dataset, dataset_size, batch_size):
        dataset.reset()
        X, Y = dataset.next()  # Initialize X and Y with first images
        # x=np.concatenate([dataset.next()[0] for i in tqdm(range(dataset.__len__()))])
        # y=np.concatenate([dataset.next()[1] for i in tqdm(range(dataset.__len__()))])

        for i in tqdm(range(dataset_size - 1)):
            imgages, targets = dataset.next()
            X = np.concatenate((X, imgages), axis=0)
            Y = np.concatenate((Y, targets), axis=0)
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
        return X, Y

    # Load Dataset Without Image augmentation
    def load_Dataset(self, image_size):
        train_data_gen = ImageDataGenerator()
        training_dir = self.dataset_folder
        batch_size = 8
        print("Extracting data from dataset at: " + training_dir)
        train_data = train_data_gen.flow_from_directory(directory=training_dir,
                                                        target_size=(96, 96),  # TODO change to image_size
                                                        color_mode='rgb',
                                                        classes=None,  # can be set to labels
                                                        class_mode='categorical',
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=self.seed)
        # TODO SEE HOW RETRIVE DATASET SIZE FROM TRAIN_DATA
        return self.convert_dataset_to_numpy(train_data, train_data.__len__(), batch_size)

    '''
        Return Xtrain,X_val,X_test,Ytrain,Y_test,Y_val

        specify the split for test and validation and specify the normalization mode 
        (see self.normalize_data for the modality available)
    '''

    def split_and_normalize(self, X, Y, split_test=.1, split_val=.1, normalization_mode=1):
        # Split Training and Testing
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=self.seed,
                                                            test_size=int(split_test * X.shape[0]), stratify=Y)

        # Normalize data
        X_train, X_test = self.normalize_data(X_train, X_test, 1)

        # Split Training and Validation
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, random_state=self.seed,
                                                          test_size=int(split_val * X_train.shape[0]), stratify=Y_train)

        return X_train, X_test, X_val, Y_train, Y_test, Y_val


    def get_class_i_vs_other(self,X,Y,i,class_i_lab=[1.0,0.0],class_other_lab=[0.0,1.0],num_of_class=2):
        #Extract Class i
        classes = np.argwhere(Y == 1) + 1
        classes = classes[:, 1]
        class_i_index = np.argwhere(classes == i)
        class_i_index = class_i_index[:, 0]

        # Extract Others
        class_not_i_index = np.argwhere(classes != i)
        class_not_i_index = class_not_i_index[:, 0]

        #Create new Label array
        new_Y = np.zeros((Y.shape[0],num_of_class))

        #Assign new labels:
        '''
        new_Y[class_i_index]    = [1.0,0.0]
        new_Y[class_not_i_index]= [0.0,1.0]
        '''
        new_Y[class_i_index]    = class_i_lab
        new_Y[class_not_i_index]= class_other_lab
        print("BANANA")

        print("Class in i: " , class_i_index.shape)
        print("Class in others: " , class_not_i_index.shape)
        return new_Y

    def get_samples_distributions(self, Y):
        classes, classes_distribution = np.unique(Y, axis=0, return_counts=True)
        classes = classes.argmax(1)
        return classes_distribution, classes_distribution

    # Plot Distribution Of data in the dataset
    def plot_samples_distribution(self, Y):
        classes, classes_distribution = self.get_samples_distributions(Y)
        y_pos = np.arange(len(classes))
        plt.bar(y_pos, classes_distribution, align='center', alpha=0.5)
        plt.xticks(y_pos, classes)
        plt.ylabel('Num of Samples')
        plt.title('Dataset Classes Distribution')

        plt.show()
        return

    # Return A slice of X,Y with only element of class_to_get {eg all element of specie 1}
    def get_slice_of_class(self, X, Y, class_to_get):
        classes = np.argwhere(Y == 1) + 1
        classes_index = np.argwhere(classes == class_to_get)
        classes_index = classes_index[:, 0]

        return X[classes_index], Y[classes_index]

    # Generate a new X,Y with augmented data of "num_of_images"
    # TODO ADD SOME PARAMETER TO CHANGE AUGMENTATION TYPE
    def apply_data_augmentation(self, X, Y, num_of_images, norm_mode=1,
                                disable_tqdm=False, rotation_range=15,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                zoom_range=0.3,
                                fill_mode="reflect",
                                horizontal_flip=True,
                                vertical_flip=True,
                                brightness_range=(0.5, 1.1),
                                seed=10,
                                featurewise_center=False,
                                samplewise_center=False,
                                featurewise_std_normalization=False,
                                samplewise_std_normalization=False,
                                ):
        # print("BB")
        X = self.denormalize(X, norm_mode)  # Denormalize
        # TODO PARAMETRIZE THIS PART
        data_generator = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            zoom_range=zoom_range,
            fill_mode=fill_mode,  # So that the fill is not strange
            brightness_range=brightness_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
        )
        i = 0
        batch_size = 32
        stop_condition = int(num_of_images / batch_size)
        # print("Generating " + str(num_of_images) + "Images")

        generator = data_generator.flow(
            X,
            Y,
            batch_size=32,
            shuffle=True,
            sample_weight=None,
            seed=seed,
            save_to_dir=None,
            save_format='png',
            ignore_class_split=False,
            subset=None,
        )

        generator.reset()

        generated = 0
        for i in tqdm(range(stop_condition), disable=disable_tqdm):
            imgages, targets = generator.next()
            # print(np.unique(targets,axis=0, return_counts=True))
            # break
            X = np.concatenate((X, imgages), axis=0)
            Y = np.concatenate((Y, targets), axis=0)
            generated += len(imgages)
        print(f"{generated} images generated")

        X = self.normalize(X, norm_mode)
        return X, Y

    def to_sum_1(self, array: np.ndarray):
        partial = array / np.min(array[np.nonzero(array)])
        return partial / partial.sum()

    def apply_data_augmentation_normalized(self, X, Y, num_of_images, disable_tqdm=False,
                                           rotation_range=15, width_shift_range=0.1,
                                           height_shift_range=0.1, zoom_range=0.3,
                                           fill_mode="reflect", brightness_range=(0.5, 1.1),
                                           horizontal_flip=True,
                                           vertical_flip=True,
                                           seed=10,
                                           featurewise_center=False,
                                           samplewise_center=False,
                                           featurewise_std_normalization=False,
                                           samplewise_std_normalization=False,
                                           ):
        classes, classes_distributions = self.get_samples_distributions(Y)
        to_equal = max(classes_distributions) - classes_distributions
        to_equal = self.to_sum_1(to_equal + (num_of_images - sum(to_equal)) / len(classes))
        return self.apply_data_augmentation_with_classes_distribution(X, Y, num_of_images,
                                                                      disable_tqdm=disable_tqdm,
                                                                      class_distribution=to_equal[::-1],
                                                                      rotation_range=rotation_range,
                                                                      width_shift_range=width_shift_range,
                                                                      height_shift_range=height_shift_range,
                                                                      zoom_range=zoom_range,
                                                                      fill_mode=fill_mode,
                                                                      brightness_range=brightness_range,
                                                                      horizontal_flip=horizontal_flip,
                                                                      vertical_flip=vertical_flip,
                                                                      seed=seed,
                                                                      featurewise_center=featurewise_center,
                                                                      samplewise_center=samplewise_center,
                                                                      featurewise_std_normalization=featurewise_std_normalization,
                                                                      samplewise_std_normalization=samplewise_std_normalization,
                                                                      )

    # Get num_of_images augmented data respecting the desired class distribution
    def apply_data_augmentation_with_classes_distribution(self, X, Y,
                                                          num_of_images,
                                                          class_distribution=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                                                          norm_mode=1, disable_tqdm=True,
                                                          rotation_range=15, width_shift_range=0.1,
                                                          height_shift_range=0.1, zoom_range=0.3,
                                                          fill_mode="reflect", brightness_range=(0.5, 1.1),
                                                          horizontal_flip=True,
                                                          vertical_flip=True,
                                                          seed=10,
                                                          featurewise_center=False,
                                                          samplewise_center=False,
                                                          featurewise_std_normalization=False,
                                                          samplewise_std_normalization=False,
                                                          ):
        # TODO FOR MORE COMPLEX NORMALIZATION TYPE WE NEED TO CHANGHE THIS
        # X = self.denormalize(X,norm_mode) #Denormalize #TODO CHECK CLEANER WAY!!

        print("Data Augmentation with data distribution")
        out_x = np.empty((0, 96, 96, 3))
        out_y = np.empty((0, Y.shape[1]))

        print("Data distribution = " + str(class_distribution))
        for i in tqdm(range(Y.shape[1]), disable=disable_tqdm):
            # print("Class: " + str(i+1))
            # Get all element of slice i
            curr_x, curr_y = self.get_slice_of_class(X, Y, i + 1)
            print(f"Class Size :  {str(curr_x.shape[0])}, generating: {int(num_of_images * class_distribution[i])}")
            # Apply Augmentation
            curr_x, curr_y = self.apply_data_augmentation(curr_x, curr_y, num_of_images * class_distribution[i],
                                                          disable_tqdm=disable_tqdm, rotation_range=rotation_range,
                                                          width_shift_range=width_shift_range,
                                                          height_shift_range=height_shift_range, zoom_range=zoom_range,
                                                          fill_mode=fill_mode, brightness_range=brightness_range,
                                                          horizontal_flip=horizontal_flip,
                                                          vertical_flip=vertical_flip,
                                                          seed=seed,
                                                          featurewise_center=featurewise_center,
                                                          samplewise_center=samplewise_center,
                                                          featurewise_std_normalization=featurewise_std_normalization,
                                                          samplewise_std_normalization=samplewise_std_normalization,
                                                          )
            # Concatenate result of class i
            out_x = np.concatenate((out_x, curr_x), axis=0)
            out_y = np.concatenate((out_y, curr_y), axis=0)

        # out_x = self.normalize(out_x,norm_mode)

        # Shuffle array
        p = np.random.permutation(out_x.shape[0])
        out_x = out_x[p]
        out_y = out_y[p]
        return out_x, out_y

    # Allow to save all images directly in numpy format, no need to load them 1 by one (fasten up the data augmentation problem)
    # TODO COMPLETE THIS FUNCTION (load function dosnt work properly)
    def load_dataset_from_numpy(self):
        # Check if dataset in numpy format is present
        check_images_numpy = os.path.isfile(os.path.join(self.numpy_dataset, "images.npy"))
        check_targets_numpy = os.path.isfile(os.path.join(self.numpy_dataset, "targets.npy"))

        if check_images_numpy and check_targets_numpy:
            # Load from numpy
            X = np.load(os.path.join(self.numpy_dataset, "images.npy"))
            Y = np.load(os.path.join(self.numpy_dataset, "targets.npy"))
            return X, Y
        else:
            # Load dataset using ImageDataGenerator
            X, Y = self.load_Dataset(10)

            # Save Numpy arrays to file
            np.save(os.path.join(self.numpy_dataset, 'images'), X)
            np.save(os.path.join(self.numpy_dataset, 'targets'), Y)
            # Return dataset
            return X, Y

    '''
        Normalize Dataset Data
        MODES:
        -1) Divide by max
        -2) Multiply by max
        -3) TODO ADD NEW METHODS FROM SLIDE
    '''

    def normalize_data(self, train, test, mode=1):  # TODO PUT AN ENUMERATION FOR THE NORMALIZATION TYPE

        train = self.normalize(train, mode)
        test = self.normalize(test, mode)

        # TODO image mean normalization, image deviation normalization etc... see slide
        return train, test

    def denormalize(self, X, mode=1):
        if mode == 1:
            # Normalize data
            X = X * 255.  # pixel value
        elif mode == 2:
            X = X / 255
        # elif mode==3:

        return X

    def normalize(self, X, mode=1):
        if mode == 1:
            # Normalize data
            X = X / 255.  # pixel value
        elif mode == 2:
            X = X * 255
        # elif mode==3:

        return X


    @staticmethod
    def to_dict(image, label):
        return {"images": image, "labels": label}

    @staticmethod
    def prepare_dataset(dataset):
        BATCH_SIZE = 128
        return (
            dataset.shuffle(10 * BATCH_SIZE)
            .map(DatasetHelper.to_dict)
            .batch(BATCH_SIZE)
        )

    @staticmethod
    def to_std_dataset(batch_dataset):
        x = []
        y = []
        batched = list(batch_dataset.as_numpy_iterator())
        for i in batched:
            x.extend(i['images'])
            y.extend(i['labels'])
        return x, y

    @staticmethod
    def to_batched_dataset(X, Y):
        return DatasetHelper.prepare_dataset(tf.data.Dataset.from_tensor_slices((X, Y)))

    @staticmethod
    def apply_keras_cv_augmentation(X, Y, layer):
        dataset = DatasetHelper.to_batched_dataset(X, Y)
        dataset = dataset.map(layer)
        return DatasetHelper.to_std_dataset(dataset)

    @staticmethod
    def keras_cv_augmentation(X, Y, rate):
        augmenters = [keras_cv.layers.MixUp(), keras_cv.layers.CutMix(),
                      keras_cv.layers.Grayscale(output_channels=3), keras_cv.layers.GridMask()]
        for augmenter in augmenters:
            rand_idx = np.random.randint(len(X), size=int(rate * len(X)))
            sub_X = X[rand_idx, :]
            sub_Y = Y[rand_idx, :]
            sub_X, sub_Y, = DatasetHelper.apply_keras_cv_augmentation(sub_X, sub_Y, augmenter)
            X = np.append(X, sub_X, axis=0)
            Y = np.append(Y, sub_Y, axis=0)

        return X, Y
