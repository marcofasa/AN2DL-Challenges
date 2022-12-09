import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler


def count_classes(dataframe, target_class):
    '''
        Count how many records there are for each possible
        value in the target_class.
    '''
    plt.figure(figsize=(17,5))
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
    plt.figure(figsize=(17,4))
    plt.plot(history['loss'], label='Training loss', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_loss'], label='Validation loss', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Categorical Crossentropy')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    plt.figure(figsize=(17,4))
    plt.plot(history['accuracy'], label='Training accuracy', alpha=.8, color='#ff7f0e')
    plt.plot(history['val_accuracy'], label='Validation accuracy', alpha=.9, color='#5a9aa5')
    plt.axvline(x=best_epoch, label='Best epoch', alpha=.3, ls='--', color='#5a9aa5')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(alpha=.3)
    plt.show()

    plt.figure(figsize=(17,4))
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