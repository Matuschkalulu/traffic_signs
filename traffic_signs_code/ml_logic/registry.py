from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input
from traffic_signs_code.params import *
import os


# Load the Model
def load_model(model_name = 'yolo_v2.pt'):
    model= keras.models.load_model(os.path.join(MODELS_PATH, model_name), compile=False)
    model.compile(loss='binary_crossentropy',
                    optimizer = Adam(learning_rate= 1e-7),
                    metrics=['accuracy'])
    return model
