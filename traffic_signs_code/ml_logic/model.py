import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from colorama import Fore, Style
from sklearn.metrics import classification_report
from traffic_signs_code.params import *


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

def resnet_model(X_train):
    print(Fore.BLUE + "\nStart layers augmentation..." + Style.RESET_ALL)
    augmentation = Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.2, 0.2),
            layers.RandomRotation(0.1)
        ])
    base_model= ResNet50(weights='imagenet', input_shape=X_train[0,:,:].shape, include_top=False)
    base_model.trainable=False

    augment_model = Sequential([
            layers.Input(shape = X_train[0,:,:].shape),
            augmentation,
            base_model,
            layers.Flatten(),
            layers.Dense(2048, activation="relu"),
            layers.Dense(1024, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="relu"),
            layers.Dense(1, activation='sigmoid')
        ])

    augment_model.compile(loss='binary_crossentropy',
                    optimizer = Adam(learning_rate= 1e-4),
                    metrics=['accuracy'])
    print('✅ Model is augmented and compiled successfully')
    return augment_model

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
        verbose=1)

    es = EarlyStopping(patience=4 ,restore_best_weights=True)
    model = initialize_base_model()
    # Fit the model on the train data
    history = model.fit(X, y, validation_split=0.3, epochs=25, batch_size=8, callbacks=[es], verbose=1)
    print(f"✅ Model trained with min val ACCURACY: {round(np.min(history.history['val_accuracy']), 2)}")
    return model, history

def train_augment(model, train_flow, X_val_preproc, y_val, epochs, batch_size, patience):
  es= EarlyStopping(monitor='val_accuracy', mode='max', restore_best_weights=True, patience=patience)
  print(Fore.BLUE + "\nTraining the model..." + Style.RESET_ALL)
  history = model.fit(train_flow,
                          batch_size=batch_size,
                          epochs = epochs,
                          callbacks = [es],
                          validation_data = (X_val_preproc, y_val))
  print('✅ Model is fitted successfully')
  return model, history

def evaluate_model(model: Model,
                        X: np.ndarray,
                        y: np.ndarray):

    """
    Evaluate trained model performance on the test dataset
    """
    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)
    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(X, y, return_dict=True)
    loss = metrics["loss"]
    accuracy = metrics["accuracy"]
    score= f"✅ Model evaluated, Accuracy: {round(accuracy, 3)}, Loss: {round(loss,3)}"
    return score

def plot_history(history, title=''):
    print(Fore.BLUE + "\nTraining model history..." + Style.RESET_ALL)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label = 'train_loss')
    ax1.plot(history.history['val_loss'], label = 'val_loss')
    ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy')
    ax2.plot(history.history['val_accuracy'], label='val accuracy')
    ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    fig.savefig(os.path.join(OUTPUT_PATH, 'model_history.png'))
    print('✅ Model history is saved to local disk successfully')
    return (ax1, ax2)

def report(model, X, y):
    print(Fore.BLUE + "\nReport the model performance..." + Style.RESET_ALL)
    y_pred= np.round(model.predict(X))
    results_df = pd.DataFrame({"actual": y,
                           "predicted": y_pred[:,0]}) #Store results in a dataframe

    confusion_matrix = pd.crosstab(index= results_df['actual'], columns= results_df['predicted'])
    print('Classification report:\n\n' ,classification_report(y, y_pred, target_names=LABELS))
    print('\nConfusion matrix:\n\n', confusion_matrix)
    print('⭐️ I am Done!')
