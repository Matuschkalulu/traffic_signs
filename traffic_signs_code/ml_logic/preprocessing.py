from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

def shuffle_data(data, target):
    X = data
    y = target
    X, y = shuffle(X, y)
    return X,y

def train_test_preproc(X,y, test_size = 0.3, random_state = 42):
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state, shuffle=True)

    X_train = tf.convert_to_tensor(X_train)
    X_test = tf.convert_to_tensor(X_test)
    y_train = np.array(y_train).astype(int)
    y_test = np.array(y_test).astype(int)

    return X_train, X_test, y_train, y_test
