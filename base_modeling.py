import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os, cv2, glob
#from google.colab import drive


# If you are working in Colab, uncomment google-colab import and the couple of lines below
#from google.colab import drive
#drive.mount('/content/drive')

# Initialize the dataset
def create_dataset(data_path):
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
    return img_data_array, class_name

# Split the data and preprocess it
def split_preprocess(X,y, data_path):
    X, y= create_dataset(data_path)
    X, y = shuffle(X, y)
    X= np.array(X)
    y= np.array(y)
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=True)
    y_train= y_train.astype(int)
    y_test= y_test.astype(int)
    X_train_preproc= preprocess_input(X_train).astype('float')
    X_test_preproc= preprocess_input(X_test).astype('float')
    return X_train_preproc, X_test_preproc, y_train, y_test

# Base Model Training
def vgg_base_model():
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

def train_model(model, X, y, validation_split, batch_size, patience):
  es= EarlyStopping(patience=patience, restore_best_weights=True, monitor= 'val_accuracy')
  history= model.fit(X, y, validation_split=validation_split, batch_size=batch_size, epochs=50, callbacks=[es],\
                  verbose=1)
  return model, history

# History Visualization
def plot_history(history):
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
    return (ax1, ax2)

# Evaluate the trained model
def model_evaluate(model, X, y):
  score= model.evaluate(X, y)[1]
  return f'Test score= {score:.2f}'

# Augmentation function to reduce overfiting
def data_augment(X_train_preproc, y_train, X_test_preproc, y_test,num, batch_size):
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
  return train_datagen, X_train_aug, y_train_aug, X_val, y_val, X_test_preproc, y_test

def train_augment(model_aug, train_datagen, X_train_aug, y_train_aug, X_val, y_val, batch_size, patience):
  train_flow = train_datagen.flow(X_train_aug, y_train_aug, batch_size = batch_size)
  es = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",factor=0.1,patience=patience,verbose=2
                                                      ,mode="max",min_delta=0.0001,cooldown=0,min_lr=0, momentum= 0.9)

  history_aug = model_aug.fit(train_flow,
                          epochs = 50,
                          callbacks = [es],
                          validation_data = (X_val, y_val))

  return model_aug, history_aug

if __name__== '__main__':
    img_folder = '/data/train_all'
    X, y= create_dataset(img_folder)
    X_train_preproc, X_test_preproc, y_train, y_test= split_preprocess(X,y, img_folder)
    model= vgg_base_model()
    model, history= train_model(model, X_train_preproc, y_train, 0.3, 32, 2)
    plot_history(history)
    base_score= model_evaluate(model, X_test_preproc, y_test)

    train_datagen, X_train_aug, y_train_aug, X_val, y_val, X_test, y_test= data_augment(X_train_preproc, y_train, X_test_preproc, y_test, 0.80, 64)
    model_aug= vgg_base_model()
    model_aug, history_aug= train_augment(model_aug, train_datagen, X_train_aug, y_train_aug, X_val, y_val, 64, 2)
    plot_history(history_aug)
    aug_score= model_evaluate(model_aug, X_test_preproc, y_test)

    # Save the model
    #model_path= '/models/model.h5'
    #model_aug.save(model_path)
