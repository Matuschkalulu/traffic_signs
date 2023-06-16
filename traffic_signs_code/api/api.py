import pandas as pd
import numpy as np
import tensorflow as tf
import colorama
from colorama import Fore, Style
import cv2
#from traffic_signs_code.ml_logic import load_model
#from traffic_signs_code.ml_logic import pred
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict():
    """
    Takes in images
    Crops the images as needed
    Make a single image prediction.
    """
    crop_img= cv2.imread('example13.jpeg', cv2.IMREAD_COLOR)
    crop_img= cv2.resize(crop_img, (100,100), interpolation = cv2.INTER_AREA).astype('float')
    crop_img = np.array(crop_img)/255.
    crop_img= np.expand_dims(crop_img, axis=0)
    print(crop_img.shape)
    model= tf.keras.models.load_model('model_20230615_final.h5', compile=False)
    model.compile(loss='binary_crossentropy',
                optimizer = 'adam',
                metrics=['accuracy'])
    pred= model.predict(crop_img)
    if pred.astype('float32') > 0.6:
        return print(Fore.YELLOW + f"\nThis is an unrecognizable sign with a prediction value: {pred}" + Style.RESET_ALL)
    else:
        return print(Fore.GREEN + f"\nThis is an recognizable sign with a prediction value: {pred}" + Style.RESET_ALL)


@app.get("/")
def root():
    return {'greeting': 'Hello'}
