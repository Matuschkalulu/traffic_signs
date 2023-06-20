import pandas as pd
import numpy as np
import colorama
from colorama import Fore, Style
import cv2
#from traffic_signs_code.ml_logic import load_model
#from traffic_signs_code.ml_logic import pred
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
from keras.applications.vgg16 import preprocess_input
from traffic_signs_code.ml_logic.miscfunc  import load_model

from PIL import Image
import io

app = FastAPI()
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/ImagePrediction")
async def create_prediction(file: UploadFile = File(...) ):#UploadFile = File(...)
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    # Decode the numpy array as an image
    pred_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    print("Image read: ",pred_image)
    imgarray = np.array(pred_image)
    print(imgarray.shape)
    image=cv2.resize(imgarray, (224, 224),interpolation = cv2.INTER_AREA)
    image=np.array(image)
    image = np.expand_dims(image,axis = 0)
    img_preprocessed = preprocess_input(image)
    print("Preprocessed Image", img_preprocessed.shape)

    model=load_model()
    print("Model Loaded")
    model.compile(loss='binary_crossentropy',
                  optimizer ='adam',
                  metrics=['accuracy'])
    print("Model Compile Done")
    y_pred= model.predict(img_preprocessed)
    print(y_pred)
    if y_pred.astype('float32') > 0.4:
        return {'Value': float(y_pred)}
    else:
        return {'Value': float(y_pred)}



@app.get("/")
def root():
    return {'greeting': 'Hello'}
