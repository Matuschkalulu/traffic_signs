from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os, cv2, glob

# Uncomment the couple of lines below if using colab
#from google.colab import drive
#drive.mount('/content/drive')

 # Load Data
def create_dataset(img_folder):
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
            image = image.astype('float')
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

# Split the data and preprocess it
def split_preprocess(X,y):
    X, y = shuffle(X, y)
    X= np.array(X)
    y= np.array(y)

    train_size= int(len(X) *0.60)
    val_size= train_size + int(len(X) * 0.2)
    X_train= X[:train_size]
    y_train= y[:train_size]
    X_val= X[train_size:val_size]
    y_val= y[train_size:val_size]
    X_test= X[val_size:]
    y_test= y[val_size:]

    X_train_preproc= preprocess_input(X_train).astype('float')
    X_val_preproc= preprocess_input(X_val).astype('float')
    X_test_preproc= preprocess_input(X_test).astype('float')
    y_train= y_train.astype('float')
    y_val= y_val.astype('float')
    y_test= y_test.astype('float')
    return X_train_preproc, X_val_preproc, X_test_preproc, y_train, y_val, y_test

# VGG Model Training
def vgg_base_model():
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

# Augmentation function
def data_augment(X_train_preproc, y_train, batch_size):
  def custom_augmentation(np_tensor):

    def random_contrast(np_tensor):
        return tf.image.random_contrast(np_tensor, 0.3, 1.2)

    def random_saturation(np_tensor):
        return tf.image.random_saturation(np_tensor, 0.3, 1.2)

    def random_hue(np_tensor):
        return tf.image.random_hue(np_tensor, 0.2)

    def gaussian_noise(np_tensor):
        mean = 0
        # variance: randomly between 1 to 25
        var = np.random.randint(1, 26)
        # sigma is square root of the variance value
        noise = np.random.normal(mean,var**0.5,np_tensor.shape)
        return np.clip(np_tensor + noise, 0, 255).astype('float')

    augmnted_tensor = random_contrast(np_tensor)
    augmnted_tensor = random_saturation(augmnted_tensor)
    augmnted_tensor = random_hue(augmnted_tensor)
    augmnted_tensor = gaussian_noise(augmnted_tensor)
    return np.array(augmnted_tensor)

  train_datagen = ImageDataGenerator(featurewise_center = False,
    featurewise_std_normalization = False,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True, shear_range=0.1,
    zoom_range = (0.8, 1.2), brightness_range=(0.8,1.2), preprocessing_function= custom_augmentation)

  train_datagen.fit(X_train_preproc)

  train_flow = train_datagen.flow(X_train_preproc, y_train, batch_size = batch_size)
  return train_flow

# TRAIN AUGMENTED DATA
def train_augment(model, train_flow, X_val_preproc, y_val, batch_size, patience):
  es= tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", restore_best_weights=True,factor=0.1,patience=patience,verbose=2
                                                      ,mode="min",min_delta=0.0001,cooldown=0,min_lr=0)

  history = model.fit(train_flow,
                          batch_size=batch_size,
                          epochs = 2,
                          callbacks = [es],
                          validation_data = (X_val_preproc, y_val))

  return model, history

# Visualize Training
def plot_history(history, title=''):
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

# Evaluate Model
def model_evaluate(model, X, y):
  score= model.evaluate(X, y)[1]
  return f'Test score= {score:.2f}'

# Predict on Test Data
def predict(X_test_preproc, y_test):
    y_pred= np.round(model.predict(X_test_preproc))
    target_names=['class_0', 'class_1'] # class_0= readable, class_1= unreadable
    print(classification_report(y_test, y_pred, target_names=target_names))
    #cm= confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show();

# Test the model performance on any test data
def test_model(test_path):
    for img in os.listdir(test_path):
        img = image.load_img(test_path + img, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_batch)
        prediction = model.predict(img_preprocessed)
        plt.imshow(img)
        plt.show()
        print(prediction)

if __name__== '__main__':
    img_folder = '/home/maly/code/Matuschkalulu/traffic_signs/raw_data/croped_train_data/'
    X, y= create_dataset(img_folder)
    X_train_preproc, X_val_preproc, X_test_preproc, y_train, y_val, y_test= split_preprocess(X,y)

    model= vgg_base_model()
    train_flow= data_augment(X_train_preproc, y_train, 32)

    aug_model= vgg_base_model()
    model, history= train_augment(aug_model, train_flow, X_val_preproc, y_val, 32, 10)
    plot_history(history)

    score= model_evaluate(model, X_test_preproc, y_test)
    predict(X_test_preproc, y_test)
    test_path= '/home/maly/code/Matuschkalulu/traffic_signs/test_images/'
    test_model(test_path)

    model_path= '/home/maly/code/Matuschkalulu/traffic_signs/models/improved_model.h5'
    model.save(model_path)
