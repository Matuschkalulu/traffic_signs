import glob
import os
import time
import pickle
import tensorflow as tf
from colorama import Fore, Style
from tensorflow.keras.optimizers import Adam
from traffic_signs_code.params import *

def load_model():
    """
    Return a saved model:
    Return None (but do not Raise) if no model is found
    """
    local_model_directory = os.path.join(LOCAL_MODEL_PATH, 'first_model')
    local_model_paths = glob.glob(f"{local_model_directory}/*")
    
    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)
    latest_model = tf.keras.models.load_model(local_model_paths, compile=False)
    latest_model.compile(loss='binary_crossentropy',
                optimizer = Adam(learning_rate= 1e-4),
                metrics=['accuracy'])
    print(":white_check_mark: Model loaded from local disk")
    return latest_model
