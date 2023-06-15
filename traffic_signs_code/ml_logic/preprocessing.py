from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow
from sklearn.utils import shuffle
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def shuffle_data(data, target):
    X = data
    y = target
    X, y = shuffle(X, y)
    return X,y

def train_test_preproc(X,y, test_size = 0.3, random_state = 42, model_selection = 'Base'):

    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, shuffle=True)

    X_train = tensorflow.convert_to_tensor(X_train)
    X_test = tensorflow.convert_to_tensor(X_test)
    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    if model_selection == 'VGG':
        X_train= preprocess_input(X_train).astype('float')
        X_test= preprocess_input(X_test).astype('float')

    print("\N{white heavy check mark}" +" Data was sucessfully split and preprocessed")

    return X_train, X_test, y_train, y_test

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

    print("\N{white heavy check mark}" +" Data was sucessfully augumented")

    return train_datagen, X_train_aug, y_train_aug, X_val, y_val, X_test_preproc, y_test
