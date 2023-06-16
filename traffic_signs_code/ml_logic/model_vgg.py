import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model, models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from colorama import Fore, Style
from tensorflow.keras.applications.vgg16 import VGG16

from traffic_signs_code import params
from traffic_signs_code.ml_logic import data , preprocessing, model


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

def train_VGG_model(model, X, y, validation_split, batch_size, patience):
    es= EarlyStopping(patience=patience, restore_best_weights=True, monitor= 'val_accuracy')
    history= model.fit(X, y, validation_split=validation_split, batch_size=batch_size, epochs=50, callbacks=[es],\
                  verbose=1)
    return model, history

def train_VGG_augment(model_aug, train_datagen, X_train_aug, y_train_aug, X_val, y_val, batch_size, patience):
  train_flow = train_datagen.flow(X_train_aug, y_train_aug, batch_size = batch_size)
  es = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",factor=0.1,patience=patience,verbose=2
                                                      ,mode="max",min_delta=0.0001,cooldown=0,min_lr=0, momentum= 0.9)

  history_aug = model_aug.fit(train_flow,
                          epochs = 50,
                          callbacks = [es],
                          validation_data = (X_val, y_val))
  print(f"✅ VGG-Model evaluated, Accuracy: {round(accuracy, 3)}")
  return model_aug, history_aug

def model_VGG_evaluate(model, X, y):
  score= model.evaluate(X, y)[1]
  return f'Test score= {score:.3f}'

if __name__== '__main__':
    X, y= data.create_dataset('VGG')
    X,y = preprocessing.shuffle_data(X,y)
    X_train_preproc, X_test_preproc, y_train, y_test= preprocessing.train_test_preproc(X,y)

    #Processing using VGG Model

    print(Fore.YELLOW + f"\nProcessing using VGG Model..." + Style.RESET_ALL)
    model_vgg= initialize_VGG_model()
    model_vgg, history= train_VGG_model(model_vgg, X_train_preproc, y_train, 0.3, 32, 2)
    model.plot_history(history)
    vgg_model_score= model_VGG_evaluate(model_vgg, X_test_preproc, y_test)

    #Processing using Data Augmentation and VGG Model

    print(Fore.MAGENTA + f"\nProcessing using Data Augmentation & VGG Model..." + Style.RESET_ALL)
    train_datagen, X_train_aug, y_train_aug, X_val, y_val, X_test, y_test= preprocessing.data_augment(X_train_preproc, y_train, X_test_preproc, y_test, 0.80, 64)
    model_aug= initialize_VGG_model()
    model_aug, history_aug= train_VGG_augment(model_aug, train_datagen, X_train_aug, y_train_aug, X_val, y_val, 64, 2)
    model.plot_history(history_aug)
    aug_VGG_score= model_VGG_evaluate(model_aug, X_test_preproc, y_test)
