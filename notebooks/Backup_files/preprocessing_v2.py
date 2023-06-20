import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.utils import shuffle

def split_process_data(train_data, test_data):
    X = []
    y = []
    X_test=[]
    y_test=[]
    for feature, label in train_data:
        X.append(feature)
        y.append(label)
    for feature, label in test_data:
        X_test.append(feature)
        y_test.append(label)

    X, y = shuffle(X, y)
    X_test, y_test = shuffle(X_test, y_test)
    X, y=np.array(X), np.array(y)
    X_test, y_test=np.array(X_test), np.array(y_test)

    train_size= int(len(X) *0.80)
    X_train= X[:train_size]
    y_train= y[:train_size]

    X_val= X[train_size:]
    y_val= y[train_size:]

    X_train_preproc= preprocess_input(X_train).astype('float')
    X_val_preproc= preprocess_input(X_val).astype('float')
    X_test_preproc= preprocess_input(X_test).astype('float')

    y_train= y_train.astype('float')
    y_val= y_val.astype('float')
    y_test= y_test.astype('float')
    return X_train_preproc, X_val_preproc, X_test_preproc, y_train, y_val, y_test
