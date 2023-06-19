import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, mean_squared_error
from traffic_signs_code.params import *


def train_augment(model, train_flow, X_val_preproc, y_val, batch_size, patience):
  es= EarlyStopping(monitor="val_accuracy", mode='max', restore_best_weights=True,patience=patience,verbose=2)
  history = model.fit(train_flow,
                          batch_size=batch_size,
                          epochs = 50,
                          callbacks = [es],
                          validation_data = (X_val_preproc, y_val))

  return model, history

def predict(model_vgg, X_test_preproc, y_test):
    y_pred= np.round(model_vgg.predict(X_test_preproc))
    print(classification_report(y_test, y_pred, target_names=labels))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()
    print("\N{white heavy check mark}" + " Confusion Matrix sucessfully created")

def model_VGG_evaluate(model_vgg, X, y_test, y_pred):
  score= model_vgg.evaluate(X, y_test)[1]
  mse= mean_squared_error(y_test, y_pred)
  return f'Test score= {score:.3f} \n mse= {mse:.3f}'
