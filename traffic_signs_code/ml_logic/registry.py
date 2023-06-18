from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input
from traffic_signs_code.params import *


# Load the Model
def load_model():
    model= keras.models.load_model(VGG_Model_path, compile=False)
    model.compile(loss='binary_crossentropy',
                    optimizer = Adam(learning_rate= 1e-7),
                    metrics=['accuracy'])
    return model
