import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tsaug
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from tsaug import *

training_labels = {
    "Wish": 0,
    "Another": 1,
    "Comfortably": 2,
    "Money": 3,
    "Breathe": 4,
    "Time": 5,
    "Brain": 6,
    "Echoes": 7,
    "Wearing": 8,
    "Sorrow": 9,
    "Hey": 10,
    "Shine": 11,
}


def split_train_test_timeseries(dataset, test_size=None):
    if test_size is None:
        test_size = 0.8 * len(dataset)
    X_train_raw = dataset.iloc[:-test_size]
    # y_train_raw = y.iloc[:-test_size]
    X_test_raw = dataset.iloc[-test_size:]
    # y_test_raw = y.iloc[-test_size:]
    # print(X_train_raw.shape, X_test_raw.shape)

    # Normalize both features and labels
    X_min = X_train_raw.min()
    X_max = X_train_raw.max()

    X_train_raw = (X_train_raw - X_min) / (X_max - X_min)
    X_test_raw = (X_test_raw - X_min) / (X_max - X_min)
    return X_train_raw, X_test_raw


def load_data(path_x="data/x_train.npy", path_y="data/y_train.npy"):
    return np.load(path_x), np.load(path_y)


def reconstruct_data(x, y):
    X_first = x[0]
    df = pd.DataFrame()
    for i in tqdm(range(x.shape[0])):
        X_first = pd.DataFrame(x[i])
        X_first["y"] = y[i]
        # X_first["id"]=i
        df = df.append(X_first)
    return df


def build_sequences(df, window=36, stride=36):
    # Sanity check to avoid runtime errors (stride used for limited computing power)
    assert window % stride == 0
    dataset = []
    labels = []
    for id in df['id'].unique():
        # Take only meaningful features
        temp = df[df['id'] == id][['x_axis', 'y_axis', 'z_axis']].values
        # Save the label
        label = df[df['id'] == id]['activity'].values[0]
        # Compute padding length
        padding_len = window - len(temp) % window
        # Create padding and concatenate it -> when you arrive at the end the window will overflow the series so you pad  the window len - the rest of the array/window in order to have perfect window jumps
        padding = np.zeros((padding_len, 3), dtype='float64')
        temp = np.concatenate((temp, padding))
        # Build features windows with their corresponging labels
        idx = 0
        while idx + window <= len(temp):
            dataset.append(temp[idx:idx + window])
            labels.append(label)
            idx += stride  # starts from zer0,50,..,
    dataset = np.array(dataset)
    labels = np.array(labels)
    return dataset, labels


def count_classes(dataframe, target_class):
    '''
        Count how many records there are for each possible
        value in the target_class.
    '''
    dataframe = dataframe.appe
    plt.figure(figsize=(17, 5))
    sns.countplot(x=target_class, data=dataframe, order=dataframe[target_class].value_counts().index)
    plt.title("{}".format(target_class))
    plt.show()


def inspect_class(df, target_class, class_, attrs=[]):
    data = df[df[target_class] == class_][attrs][:500]
    axis = data.plot(subplots=True, figsize=(17, 9), title=class_)
    for ax in axis:
        ax.legend(loc='lower right')


'''
One interesting feature of random forests is that they return an 
index of feature importance that can be used both to have a better 
understanding of what influence the target values and but also for 
feature selection.
'''


def PlotRFFeatureImportance(X, forest_model, feature_names, sort_importance=True):
    '''
        Print and plot the features in order of their importance.
        @param X: Dataframe
        @param forest_model: Random Forest Regressor model
        @param feature_names: array of features name
        @sort_imporance: if True, importance in decreasing order
    '''
    importances = forest_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest_model.estimators_],
                 axis=0)
    if (sort_importance):
        indices = np.argsort(importances)[::-1]
    else:
        indices = np.argsort(feature_names)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d %s (%f)" % (f + 1, indices[f], feature_names[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()


def MinMaxScaler_features(df, cols_to_scale=[]):
    '''
        Scale the features applying MinMax scaler.
    '''
    scaler = MinMaxScaler()
    scaler = scaler.fit(df[cols_to_scale])
    df.loc[:, cols_to_scale] = scaler.transform(df[cols_to_scale].to_numpy())

    return df


def plot_performances(history):
    '''
        Plot losses and accuracy performed by the model.
    '''
    best_epoch = np.argmax(history['val_accuracy'])
    plt.figure(figsize=(17, 4))
    plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Categorical Crossentropy')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    plt.figure(figsize=(17, 4))
    plt.plot(history['accuracy'], label='Training accuracy', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_accuracy'], label='Validation accuracy', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    plt.figure(figsize=(17, 4))
    plt.plot(history['lr'], label='Learning Rate', alpha=.8, color='#ff7f0e')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()


def confusionmatr_and_metrics(model, X_test, y_test, label_mapping):
    '''
        Plot the confusion matrix and print some 
        classifiation metrics.
    '''
    predictions = model.predict(X_test)
    # Compute the confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1))

    # Compute the classification metrics
    accuracy = accuracy_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1))
    precision = precision_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
    recall = recall_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
    f1 = f1_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
    print('Accuracy:', accuracy.round(4))
    print('Precision:', precision.round(4))
    print('Recall:', recall.round(4))
    print('F1:', f1.round(4))

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm.T, cmap='Blues', xticklabels=list(label_mapping.keys()), yticklabels=list(label_mapping.keys()))
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()


def plot_aug(x, x_aug):
    # color red for original and green for augmented
    plt.plot(x, color='red')
    plt.plot(x_aug, color='green')

def data_smoothing(data, window_size=10):
    '''
        Apply a moving average to the data.
    '''
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def random_cropping(x, y, crop_size=100):
    '''
        Randomly crop the data.
    '''
    start = np.random.randint(0, x.shape[0]-crop_size)
    end = start + crop_size
    return x[start:end], y[start:end]

# Random Rotation: Rotate the data series to create new data points.
def random_rotation(x, y, max_rotation=10):
    '''
        Randomly rotate the data.
    '''
    rotation = np.random.randint(-max_rotation, max_rotation)
    return rotate(x, rotation), y


# Random Flipping: Flip the data series horizontally or vertically to create new data points.
def random_flipping(x, y, max_flip=1):
    '''
        Randomly flip the data.
    '''
    flip = np.random.randint(-max_flip, max_flip)
    return np.flip(x, axis=flip), y

# Random Scaling: Scale the data series up or down to create new data points.
def random_scaling(x, y, max_scale=1.5):
    '''
        Randomly scale the data.
    '''
    scale = np.random.uniform(1, max_scale)
    return scale*x, y


def apply_aug_function(x, y, aug_function, dim=100, **kwargs):
    '''
        generate new dim data series by taking random dataseries in x and applying aug_function
    '''
    x_aug = np.zeros((dim, x.shape[1]))
    y_aug = np.zeros((dim, y.shape[1]))
    for i in range(dim):
        idx = np.random.randint(0, x.shape[0])
        x_aug[i], y_aug[i] = aug_function(x[idx], y[idx], **kwargs)
    return x_aug, y_aug

