import pandas as pd
import numpy as np
import tensorflow as tf
import colorama
from colorama import Fore, Style
import cv2
#from traffic_signs_code.ml_logic import load_model
#from traffic_signs_code.ml_logic import pred
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from traffic_signs_code.ml_logic.miscfunc import load_model
import matplotlib.pyplot as plt
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

@app.post("/ImagePrediction/")
async def create_prediction(file: UploadFile = File(...)):
    contents = await file.read()
    pred_image = Image.open(io.BytesIO(contents))
    print("Image read: ",pred_image)
    imgarray = np.array(pred_image)
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
        return print(Fore.YELLOW + f"\nThis is an unrecognizable sign with a prediction value: {y_pred}" + Style.RESET_ALL)
    else:
        return print(Fore.GREEN + f"\nThis is an recognizable sign with a prediction value: {y_pred}" + Style.RESET_ALL)



@app.get("/")
def root():
    return {'greeting': 'Hello'}
