from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error
from sklearn.metrics import confusion_matrix
from PIL import Image
from tensorflow import keras


class Utils:
    def create_seed(tf,seed=42):
        # Random seed for reproducibility
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.compat.v1.set_random_seed(seed)

    def monitor(histories, names, colors, early_stopping=1):
        assert len(histories) == len(names)
        assert len(histories) == len(colors)
        plt.figure(figsize=(15,6))
        for idx in range(len(histories)):
            plt.plot(histories[idx]['mse'][:-early_stopping], label=names[idx]+' Training', alpha=.4, color=colors[idx], linestyle='--')
            plt.plot(histories[idx]['val_mse'][:-early_stopping], label=names[idx]+' Validation', alpha=.8, color=colors[idx])
        plt.ylim(0.0075, 0.02)
        plt.title('Mean Squared Error')
        plt.legend(bbox_to_anchor=(1,1))
        plt.grid(alpha=.3)
        plt.show()

    def plot_residuals(model, X_, y_):
        X_['sort'] = y_
        X_ = X_.sort_values(by=['sort'])
        y_ = np.expand_dims(X_['sort'], 1)
        X_.drop(['sort'], axis=1, inplace=True)

        y_pred = model.predict(X_, verbose=0)
        MSE = mean_squared_error(y_,y_pred)

        print('Mean Squared Error (MSE): %.4f' % MSE)

        mpl.rcParams.update(mpl.rcParamsDefault)
        sns.set(font_scale=1.1, style=None, palette='Set1')
        plt.figure(figsize=(15,5))
        plt.scatter(np.arange(len(y_pred)), y_pred, label='Prediction', color='#1f77b4')
        plt.scatter(np.arange(len(y_)), y_, label='True', color='#d62728')

        for i in range(len(y_)):
            if(y_[i]>=y_pred[i]):
                plt.vlines(i,y_pred[i],y_[i],alpha=.5)
            else:
                plt.vlines(i,y_[i],y_pred[i],alpha=.5)

        plt.legend()
        plt.grid(alpha=.3)
        plt.show()

    def show_prediction(X_test,Y_test,prediction_index,labels,predicted_vector):
        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(15,5)
        ax1.imshow(X_test[prediction_index])
        #Each class has its own score
        #We select the label with the largest prediction score
        ax1.set_title('True label: '+labels[np.argmax(Y_test[prediction_index])+1])
        ax2.barh(list(labels.values()), predicted_vector[prediction_index], color=plt.get_cmap('Paired').colors)
        ax2.set_title('Predicted label: '+labels[np.argmax(predicted_vector[prediction_index])])
        ax2.grid(alpha=.3)
        plt.show()

    def save_model(model,name="Model"):
        model.save(name)

    def load_model(name,relative_path=''):
        return keras.models.load_model(relative_path+name)

    def normalize_data(train,test,mode=1):
        if mode==1:
            # Normalize data
            train = train/255. #pixel value
            test = test/255. #pixel value
        elif mode==2:
            train = train*255. #pixel value
            test = test*255. #pixel value
        #elif mode==3:


        return train,test


    def plot_latent_filters(tfk,X_train,model, layers, image):
        fig, axes = plt.subplots(1, len(layers), figsize=(20,5))
        for j,layer in enumerate(layers):
            ax = axes[j]
            ax.imshow(tfk.Sequential(model.layers[:layer]).predict(tf.expand_dims(image,axis=0), verbose=0)[0,:,:,0], cmap='gray')
        plt.show()

        layers = [1,2,3,4,5,6,7]
        n = 4

        for i in range(n):
            tfk.plot_latent_filters(model, layers, X_train[random.randint(0, len(X_train))])

    def show_confusion_matrix(labels,predictions,y_test):
        # Build the confusion matrix (using scikit-learn)
        cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1))

        # Compute the classification metrics using the confusion matrix builded
        accuracy = accuracy_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1))
        precision = precision_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1), average='macro') # macro-> maetric for each class and the compute avg
        recall = recall_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
        f1 = f1_score(np.argmax(y_test, axis=-1), np.argmax(predictions, axis=-1), average='macro')
        print('Accuracy:',accuracy.round(4))
        print('Precision:',precision.round(4))
        print('Recall:',recall.round(4))
        print('F1:',f1.round(4))

        # Plot the confusion matrix
        plt.figure(figsize=(10,8))
        sns.heatmap(cm.T, xticklabels=list(labels.values()), yticklabels=list(labels.values()))
        plt.xlabel('True labels')
        plt.ylabel('Predicted labels')
        plt.show()

    def plot_phase_train_vs_validation(history,train_error_name='loss',val_error_name='accuracy',model=None,X_train=None,y_train=None,X_val=None, y_val=None):
        # Plot the training
        plt.figure(figsize=(20,5))
        plt.plot(history[train_error_name], label='Training', alpha=.8, color='#ff7f0e')
        plt.plot(history['val_'+train_error_name], label='Validation', alpha=.8, color='#4D61E2')
        plt.legend(loc=train_error_name)
        plt.title(train_error_name)
        plt.grid(alpha=.3)

        plt.figure(figsize=(20,5))
        plt.plot(history[val_error_name], label='Training', alpha=.8, color='#ff7f0e')
        plt.plot(history['val_'+val_error_name], label='Validation', alpha=.8, color='#4D61E2')
        plt.legend(loc='upper left')
        plt.title(val_error_name)
        plt.grid(alpha=.3)

        plt.show()

        if model!=None:
            print('Train Performance')
            history.plot_residuals(model, X_train.copy(), y_train.copy())
            print('Validation Performance')
            history.plot_residuals(model=model, X_= X_val.copy(), Y_= y_val.copy())

    def inspect_data(labels,X_train_val,y_train_val):
        # Inspect the data
        num_row = 2
        num_col = 5
        fig, axes = plt.subplots(num_row, num_col, figsize=(10*num_row,2*num_col))
        for i in range(num_row*num_col):
            ax = axes[i//num_col, i%num_col]
            ax.imshow(X_train_val[i])
            ax.set_title('{}'.format(labels[y_train_val[i][0]]))
        plt.tight_layout()
        plt.show()
        # Inspect the target
        # Distribution
        plt.figure(figsize=(15,5))
        sns.histplot(data=pd.DataFrame(y_train_val, columns=['digit']), x='digit', stat="percent", element="step", fill=False, kde=True)
        plt.show()

        print('Counting occurrences of target classes:')
        print(pd.DataFrame(y_train_val, columns=['digit'])['digit'].value_counts())