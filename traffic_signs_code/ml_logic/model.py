import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model, models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from colorama import Fore, Style
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os, cv2, glob

from traffic_signs_code import params
from traffic_signs_code.ml_logic import data , preprocessing
#End of Import

""" def create_dataset(data_path):
    img_data_array=[]
    class_name=[]
    IMG_WIDTH=224
    IMG_HEIGHT=224
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name """

""" def split_preprocess(X,y, data_path):
    X, y= create_dataset(data_path)
    X, y = shuffle(X, y)
    X= np.array(X)
    y= np.array(y)
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)
    y_train= y_train.astype(int)
    y_test= y_test.astype(int)
    X_train_preproc= preprocess_input(X_train).astype('float')
    X_test_preproc= preprocess_input(X_test).astype('float')
    return X_train_preproc, X_test_preproc, y_train, y_test """


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

def initialize_VGG_model():
    base_model= VGG16(weights='imagenet', input_shape=X_train_preproc[0,:,:].shape, include_top=False)
    base_model.trainable=False
    last_layer= base_model.output

    x= Flatten()(last_layer)
    x= Dense(1024, activation= 'relu')(x)
    x= Dense(512, activation= 'relu')(x)
    x= Dense(10, activation= 'relu')(x)
    output= Dense(1, activation= 'sigmoid')(x)

    model= Model(base_model.input, output)

    model.compile(loss='binary_crossentropy',
                optimizer = Adam(learning_rate= 1e-4),
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


def train_VGG_model(model, X, y, validation_split, batch_size, patience):
    es= EarlyStopping(patience=patience, restore_best_weights=True, monitor= 'val_accuracy')
    history= model.fit(X, y, validation_split=validation_split, batch_size=batch_size, epochs=50, callbacks=[es],\
                  verbose=1)
    return model, history


def evaluate_base_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ):

    """
    Evaluate trained model performance on the dataset
    """
    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )
    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, Accuracy: {round(accuracy, 2)}")

    return metrics

def model_VGG_evaluate(model, X, y):
  score= model.evaluate(X, y)[1]
  return f'Test score= {score:.2f}'

def train_VGG_augment(model_aug, train_datagen, X_train_aug, y_train_aug, X_val, y_val, batch_size, patience):
  train_flow = train_datagen.flow(X_train_aug, y_train_aug, batch_size = batch_size)
  es = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",factor=0.1,patience=patience,verbose=2
                                                      ,mode="max",min_delta=0.0001,cooldown=0,min_lr=0, momentum= 0.9)

  history_aug = model_aug.fit(train_flow,
                          epochs = 50,
                          callbacks = [es],
                          validation_data = (X_val, y_val))

  return model_aug, history_aug

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

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)


""" def data_augment(X_train_preproc, y_train, X_test_preproc, y_test,num, batch_size):
    train_datagen = ImageDataGenerator(featurewise_center = False,
        featurewise_std_normalization = False,
        rotation_range = 10,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True,
        zoom_range = (0.8, 1.2))

    train_datagen.fit(X_train_preproc)
    train_size= int(X_train_preproc.shape[0] *num)

    X_train_aug= X_train_preproc[:train_size].astype('float')
    y_train_aug= y_train[:train_size].astype(int)

    X_val= X_train_preproc[train_size:].astype('float')
    y_val= y_train[train_size:].astype(int)

    X_test_preproc= X_test_preproc.astype('float')
    y_test= y_test.astype(int)

    train_flow = train_datagen.flow(X_train_aug, y_train_aug, batch_size = batch_size)
    return train_datagen, X_train_aug, y_train_aug, X_val, y_val, X_test_preproc, y_test """


if __name__== '__main__':

    X, y= data.create_dataset('Base')
    X,y = preprocessing.shuffle_data(X,y)
    X_train_preproc, X_test_preproc, y_train, y_test= preprocessing.train_test_preproc(X,y)

    #Processing using Base Mode
    print(Fore.GREEN + f"\nProcessing using Base Model..." + Style.RESET_ALL)
    model1 = initialize_base_model()
    model1, history1 = train_base_model(model1, X_train_preproc, y_train)
    plot_history(history1)
    base_model_score = evaluate_base_model(model1, X_test_preproc, y_test)

    #Processing using VGG Model

    """  print(Fore.YELLOW + f"\nProcessing using VGG Model..." + Style.RESET_ALL)
    model= initialize_VGG_model()
    model, history= train_VGG_model(model, X_train_preproc, y_train, 0.3, 32, 2)
    plot_history(history)
    vgg_model_score= model_VGG_evaluate(model, X_test_preproc, y_test)

    #Processing using Data Augmentation and VGG Model

    print(Fore.MAGENTA + f"\nProcessing using Data Augmentation & VGG Model..." + Style.RESET_ALL)
    train_datagen, X_train_aug, y_train_aug, X_val, y_val, X_test, y_test= preprocessing.data_augment(X_train_preproc, y_train, X_test_preproc, y_test, 0.80, 64)
    model_aug= initialize_VGG_model()
    model_aug, history_aug= train_VGG_augment(model_aug, train_datagen, X_train_aug, y_train_aug, X_val, y_val, 64, 2)
    plot_history(history_aug)
    aug_VGG_score= model_VGG_evaluate(model_aug, X_test_preproc, y_test) """
