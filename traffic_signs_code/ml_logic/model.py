import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model, models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from colorama import Fore, Style
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import os, cv2, glob
from traffic_signs_code.params import *
from traffic_signs_code.ml_logic import data , preprocessing
#End of Import


def initialize_base_model():
    """ Initialization of the base model for traffic signs project
        Model: Sequential
        5 layers of Conv2D and Pooling
        1 layer of Flatten
        1 layer of Dense
        1 final Classification layer
    """

    model = models.Sequential()

    ### First Convolution & MaxPooling
    model.add(layers.Conv2D(8, (3,3), activation='relu', padding='same', input_shape=(100, 100, 3)))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    ### Second Convolution & MaxPooling
    model.add(layers.Conv2D(16, (3,3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    ### Third Convolution & MaxPooling
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    ### Fourth Convolution & MaxPooling
    model.add(layers.Conv2D(32, (3,3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    ### Fifth Convolution & MaxPooling
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    ### Flattening
    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation='relu'))

    ### Last layer - Classification Layer
    model.add(layers.Dense(1, activation='sigmoid'))

    ### Model compilation
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model



def train_base_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=16,
        patience=5,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ):

    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    es = EarlyStopping(patience=4 ,restore_best_weights=True)
    model = initialize_base_model()
    # Fit the model on the train data
    history = model.fit(X, y, validation_split=0.3, epochs=25, batch_size=8, callbacks=[es], verbose=1)

    print(f"✅ Model trained with min val ACCURACY: {round(np.min(history.history['val_accuracy']), 2)}")

    return model, history



def evaluate_base_model(model: Model,
                        X: np.ndarray,
                        y: np.ndarray,
                        batch_size=64):

    """
    Evaluate trained model performance on the dataset
    """
    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(x=X,
                             y=y,
                             batch_size=batch_size,
                             verbose=0,
                             # callbacks=None,
                             return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]
    print(f"✅ Base-Model evaluated, Accuracy: {round(accuracy, 3)}, Loss: {round(loss,3)}")
    return metrics


def plot_history(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label = 'train' + exp_name)
    ax1.plot(history.history['val_loss'], label = 'val' + exp_name)
    ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()
    ax1.imshow()

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)

def report(model, y_test):
    y_pred= model.predict(X_test_preproc)
    y_pred= y_pred.astype('int')
    target_names=['class_0', 'class_1']

    results_df = pd.DataFrame({"actual": y_test,
                           "predicted": y_pred[:,0]}) #Store results in a dataframe

    confusion_matrix = pd.crosstab(index= results_df['actual'],
                               columns = results_df['predicted'])
    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix)


if __name__== '__main__':

    X, y= data.create_dataset('Base')
    X,y = preprocessing.shuffle_data(X,y)
    X_train_preproc, X_test_preproc, y_train, y_test= preprocessing.train_test_preproc(X,y)

    #Processing using Base Mode
    print(Fore.GREEN + f"\nProcessing using Base Model..." + Style.RESET_ALL)
    model = initialize_base_model()
    model, history = train_base_model(model, X_train_preproc, y_train)
    plot_history(history)

    base_model_score = evaluate_base_model(model, X_test_preproc, y_test)
    report(model, y_test)
