import numpy as np


# converts the dataset x_train, y_train in n_classes datasets
# where each y_train in 1 if the class is the one of the dataset, 0 otherwise
def convert_to_n_classes(x_train, y_train, n_classes):
    """
    Converts the dataset x_train, y_train in n_classes datasets
    where each y_train in 1 if the class is the one of the dataset, 0 otherwise.
    """
    x_train_n_classes = []
    y_train_n_classes = []
    for i in range(n_classes):
        x_train_n_classes.append(x_train)
        y_train_class = np.zeros(y_train.shape[0])
        y_train_class[np.where(np.argmax(y_train, axis=1) == i)] = 1
        y_train_n_classes.append(y_train_class)
    return x_train_n_classes, y_train_n_classes


# generate a random dataset x_train, y_train with n_classes in one hot encoding
def generate_random_dataset(n_samples, n_classes, n_features, n_timesteps):
    """
    Generates a random dataset x_train, y_train with n_classes in one hot encoding.
    """
    x_train = np.random.rand(n_samples, n_timesteps, n_features)
    y_train = np.random.randint(n_classes, size=n_samples)

    return x_train, y_train


# converts to one_hot encoding
def convert_to_one_hot(y_train, n_classes):
    """
    Converts to one_hot encoding.
    """
    y_train = np.eye(n_classes)[y_train]

    return y_train
