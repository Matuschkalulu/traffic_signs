from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import tensorflow
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def shuffle_data(X,y):
    X, y = shuffle(X, y)
    X= np.array(X)
    y= np.array(y)
    return X,y

def train_test_preproc(X,y, test_size = 0.3, random_state = 42, model_selection = 'Base'):
    '''
    =======
    Return
    X_train_preproc, X_val_preproc, X_test_preproc, y_train, y_val, y_test
    '''
    if model_selection == 'VGG':

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

        print("\N{white heavy check mark}" +" Data was sucessfully split and preprocessed")

        return X_train_preproc, X_val_preproc, X_test_preproc, y_train, y_val, y_test

    if model_selection ==  'Base':
        X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, shuffle=True)
        X_train = tensorflow.convert_to_tensor(X_train)
        X_test = tensorflow.convert_to_tensor(X_test)
        y_train = np.array(y_train).astype(int)
        y_test = np.array(y_test).astype(int)
        X_train /= 255.
        X_test /= 255.

        print("\N{white heavy check mark}" +" Data was sucessfully split and preprocessed")

        return X_train, X_test, y_train, y_test

def data_augment(X_train_preproc, y_train, batch_size):

    def custom_augmentation(np_tensor):

        def random_contrast(np_tensor):
                return tensorflow.image.random_contrast(np_tensor, 0.3, 1.2)

        def random_saturation(np_tensor):
                return tensorflow.image.random_saturation(np_tensor, 0.3, 1.2)

        def random_hue(np_tensor):
                return tensorflow.image.random_hue(np_tensor, 0.2)

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
