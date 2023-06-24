from tensorflow import keras
from colorama import Fore, Style
from tensorflow.keras.optimizers import Adam
from traffic_signs_code.params import *

def load_model(model_path: str) -> keras.Model:
    """
    Return a saved model:
    """
    latest_model = keras.models.load_model(model_path, compile=False)
    latest_model.compile(loss='binary_crossentropy',
                optimizer = Adam(learning_rate= 1e-4),
                metrics=['accuracy'])
    print('✅ Model is successfully loaded and compiled from local disk')
    return latest_model

def save_model(model: keras.Model, model_path: str) -> None:
    """
    Saves the trained model to a local directory
    """
    model.save(model_path)
    print('✅ Model is successfully saved to local disk')
