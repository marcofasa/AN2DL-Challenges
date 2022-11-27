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
from datetime import datetime
import keras.backend as K


class ModelHelper:
    seed = 42
    model = None

    def __init__ (self, path, labels, create_dirs=False):
        self.name = ""
        self.path = path
        self.labels = labels   #Contain the array of labels
        self.local_checkpoints = os.path.join(self.path, 'local_checkpoints')
        self.local_tensorboard = os.path.join(self.path, 'local_tensorboard')
        #The folder where load/save models
        self.models_dir = os.path.join(self.path, 'local_saved_models')

        if create_dirs:
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir)
            if not os.path.exists(self.local_checkpoints):
                os.makedirs(self.local_checkpoints)
            if not os.path.exists(self.local_tensorboard):
                os.makedirs(self.local_tensorboard)


    def create_seed(self,tf,seed=42):
        self.seed = 42
        # Random seed for reproducibility
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        tf.compat.v1.set_random_seed(seed)

    def monitor(self,histories, names, colors, early_stopping=1):
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

    def plot_residuals(self,model, X_, y_):
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

    def show_prediction(self,X_test,Y_test,prediction_index):
        predictions = self.model.predict(X_test)

        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.set_size_inches(15,5)
        ax1.imshow(X_test[prediction_index])
        #Each class has its own score
        #We select the label with the largest prediction score
        ax1.set_title('True label: '+self.labels[np.argmax(Y_test[prediction_index])])
        ax2.barh(list(self.labels.values()), predictions[prediction_index], color=plt.get_cmap('Paired').colors)
        ax2.set_title('Predicted label: '+ self.labels[np.argmax(predictions[prediction_index])])
        ax2.grid(alpha=.3)
        plt.show()

    def show_predictions(self,X_test,Y_test,items_count=8):
        predictions = self.model.predict(X_test)

        items_showed = 0
        fig, ax = plt.subplots(ncols=4, nrows=int(items_count/4)+1, figsize=(20,20))
        for row in ax:
            for j, col in enumerate(row):
                if j%2 == 0:
                    col.imshow(X_test[items_showed])
                    col.title.set_text('True label: '+self.labels[np.argmax(Y_test[items_showed])])
                else:
                    col.barh(list(self.labels.values()), predictions[items_showed], color=plt.get_cmap('Paired').colors)
                    col.title.set_text('Predicted label: '+ self.labels[np.argmax(predictions[items_showed])])
                    items_showed += 1
                    if items_showed > items_count:
                        break
        plt.show()

    #Save model To memory
    def save_model(self,model,name="Model"):
        model.save(os.path.join(self.models_dir, name))
        self.model = model

    #Load Model from memory
    def load_model(self,name):
        model = tf.keras.models.load_model(os.path.join(self.models_dir, name))
        self.model = model
        return model

    def plot_latent_filters(self,tfk,X_train,model, layers, image):
        fig, axes = plt.subplots(1, len(layers), figsize=(20,5))
        for j,layer in enumerate(layers):
            ax = axes[j]
            ax.imshow(tfk.Sequential(model.layers[:layer]).predict(tf.expand_dims(image,axis=0), verbose=0)[0,:,:,0], cmap='gray')
        plt.show()

        layers = [1,2,3,4,5,6,7]
        n = 4

        for i in range(n):
            tfk.plot_latent_filters(model, layers, X_train[random.randint(0, len(X_train))])

    def set_model(self,model):
        print("Set Model to Model Helper")
        self.model = model

    #Calculate and show the confusion matrix of the model
    def show_confusion_matrix(self,x_test,y_test,model=None):
        if model != None:
            self.set_model(model)

        if self.model == None:
            print("No Model Loaded in this helper class, try use save_model(model,name) function")
            return None
        
        predictions = self.model.predict(x_test)

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
        sns.heatmap(cm.T, xticklabels=list(self.labels.values()), yticklabels=list(self.labels.values()))
        plt.xlabel('True labels')
        plt.ylabel('Predicted labels')
        plt.show()

    def plot_phase_train_vs_validation(self,history,train_error_name='loss',val_error_name='accuracy',model=None,X_train=None,y_train=None,X_val=None, y_val=None):
        # Plot the training
        plt.figure(figsize=(20,5))
        plt.plot(history[train_error_name], label='Training', alpha=.8, color='#ff7f0e')
        plt.plot(history['val_'+train_error_name], label='Validation', alpha=.8, color='#4D61E2')
        plt.legend(loc='upper left')
        plt.title(train_error_name)
        plt.grid(alpha=.3)
        print("BABABA")
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

    def inspect_data(self,labels,X_train_val,y_train_val):
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

    '''
    Create the callbacks function, select the desired functions by setting its parameter to 1
    1) checkPoint    -> Save model checkpoint each epoch
    2) earlyStopping -> If model is stuck with no improvement for more then [patience] restore best weights
    3) tensorboard   -> Save all training info for tensorboard visualization

    eg: callbacks = createCallbacks(checkPoint = True, save_weights_only = False,earlyStopping = True)
        history= model.fit(...., callbacks = callbacks)
    '''


    
    def createCallbacks(self, where_to_save=None, model_name=None, checkPoints=False, earlyStopping=False, tensorboard=False, patience=10, save_weights_only=True, save_best_only=False):
        """!
            Does something ...

            @param where_to_save String: path specify where to save the checkpoints (e.g., "gdrive/MyDrive/--")
            @param model_name: name of the built model
        """
        
        callbacks = []
        
        #MODEL CHECKPOINTING AT EACH EPOCH
        if checkPoints:
            # Specify where to save the checkpoints && new save format (model name + time of generation)
            if where_to_save is not None:
                exps_dir = os.path.join(where_to_save)
                if not os.path.exists(exps_dir):
                    os.makedirs(exps_dir)

                now = datetime.now().strftime('%b%d_%H-%M-%S')

                exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
                if not os.path.exists(exp_dir):
                    os.makedirs(exp_dir)

                ckpt_dir = os.path.join(exp_dir, 'ckpts')
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)

                ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp.ckpt'), 
                                                     save_weights_only=save_weights_only, # True to save only weights
                                                     save_best_only=save_best_only) # True to save only the best epoch
            else:   # "where_to_save" NOT specified
                ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.local_checkpoints, 'cp.ckpt'), 
                                                     save_weights_only=save_weights_only, # True to save only weights
                                                     save_best_only=save_best_only) # True to save only the best epoch
            callbacks.append(ckpt_callback)

        #SAVE TRAINING INFO FOR TENSORBOARD
        if tensorboard:
            # By default shows losses and metrics for both training and validation
            tb_callback = tf.keras.callbacks.TensorBoard(log_dir=self.local_tensorboard, 
                                                        profile_batch=0,
                                                        histogram_freq=1)  # if > 0 (epochs) shows weights histograms
            callbacks.append(tb_callback)
        #EARLY STOPPING IF OVERFITTING
        if earlyStopping:
            # Early Stopping
            # --------------
            es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True)
            callbacks.append(es_callback)

        return callbacks