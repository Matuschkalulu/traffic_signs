import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from colorama import Fore, Style

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

from traffic_signs_code.params import *
from traffic_signs_code.ml_logic import data , preprocessing, model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def initialize_VGG_model():
    augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomRotation(0.1)
    ])
    base_model= VGG16(weights='imagenet', input_shape=X_train_preproc[0,:,:].shape, include_top=False)
    base_model.trainable=False

    augment_model = Sequential([
        layers.Input(shape = X_train_preproc[0,:,:].shape),
        augmentation,
        base_model,
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(1, activation='sigmoid')
    ])

    augment_model.compile(loss='binary_crossentropy',
                optimizer = Adam(learning_rate= 1e-4),
                metrics=['accuracy'])

    return augment_model


def train_VGG_model(model, X, y, validation_split, batch_size, patience):
    es= EarlyStopping(patience=patience, restore_best_weights=True, monitor= 'val_accuracy')
    history= model.fit(X, y, validation_split=validation_split, batch_size=batch_size, epochs=50, callbacks=[es],\
                  verbose=1)
    return model, history

def train_augment(model, train_flow, X_val_preproc, y_val, batch_size, patience):
  es= EarlyStopping(monitor="val_accuracy", mode='max', restore_best_weights=True, patience=patience, verbose=2)

  history = model.fit(train_flow,
                          batch_size=batch_size,
                          epochs = 50,
                          callbacks = [es],
                          validation_data = (X_val_preproc, y_val))

  return model, history

def model_VGG_evaluate(model_vgg, X, y):
  score= model_vgg.evaluate(X, y)[1]
  return f'Test score= {score:.3f}'

def predict(model_vgg, X_test_preproc, y_test):
    y_pred= np.round(model_vgg.predict(X_test_preproc))
    print(classification_report(y_test, y_pred, target_names=labels))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    print("\N{white heavy check mark}" + " Confusion Matrix sucessfully created")

def test_model(test_path, model_vgg):
    for img in os.listdir(test_path):
        img = image.load_img(test_path + img, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        prediction = model_vgg.predict(img_preprocessed)
        #plt.imshow(img)
        #plt.show()
        print(prediction)
        print("\N{white heavy check mark}" + " Successfully tested")

if __name__== '__main__':
    Train_data, Test_data = data.create_dataset_with_split('VGG')
    X_train_preproc, X_val_preproc ,X_test_preproc, y_train, y_val, y_test= preprocessing.train_test_preproc(Train_data, Test_data, model_selection='VGG')

    model_vgg = initialize_VGG_model()
    train_flow= preprocessing.data_augment(X_train_preproc, y_train, 32)

    model_aug, history= train_augment(model_vgg, train_flow, X_val_preproc, y_val, 32, 10)
    #model.plot_history(history)

    score= model_VGG_evaluate(model_aug, X_test_preproc, y_test)
    predict(model_aug,X_test_preproc, y_test)
    test_model(IMG_test_path, model_vgg)

    model_aug.save(os.path.join(MODELS_path,'model_vgg_20230616'))
