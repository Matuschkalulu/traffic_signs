import pandas as pd
import numpy as np
import tensorflow as tf
import colorama
from colorama import Fore, Style
import cv2, os
from ultralytics import YOLO
#from traffic_signs_code.ml_logic import load_model
#from traffic_signs_code.ml_logic import pred
from fastapi import FastAPI
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from traffic_signs_code.ml_logic.miscfunc import load_model
from traffic_signs_code.params import *
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
    np_array = np.frombuffer(contents, np.uint8)
    pred_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    app = FastAPI()
    app.state.model= load_model()
    app.state.model.compile(loss='binary_crossentropy',
                  optimizer ='adam',
                  metrics=['accuracy'])

    custom_model = YOLO(os.path.join(LOCAL_MODEL_PATH, 'yolo_v2.pt'))
    results = custom_model(pred_image)
    pred_list=[]
    class_list=[]
    for n, box in enumerate(results[0].boxes.xywhn):
        h, w = pred_image.shape[:2]
        x1, y1, x2, y2 = box.numpy()
        x_center, y_center = int(float(x1) * w), int(float(y1) * h)
        box_width, box_height = int(float(x2) * w), int(float(y2) * h)
        x_min = int(x_center - (box_width / 2))
        y_min = int(y_center - (box_height / 2))
        crop_img= pred_image[y_min:y_min+int(box_height), x_min:x_min+int(box_width)]
        image=cv2.resize(crop_img, (160, 160),interpolation = cv2.INTER_AREA)
        image = np.expand_dims(image,axis = 0)
        img_preprocessed = preprocess_input(image)
        print("Preprocessed Image", img_preprocessed.shape)
        y_pred= app.state.model.predict(img_preprocessed)[0][0]
        pred_list.append(y_pred)
        print(y_pred)
        if y_pred < 0.6 :
            class_current= 'READABLE'
            print(Fore.YELLOW + f"\nThis is an recognizable sign with a prediction value: {y_pred}" + Style.RESET_ALL)
            #return {"Value": [float(y_pred), class_current]}
        else:
            class_current= 'UNREADABLE'
            print(Fore.GREEN + f"\nThis is an unrecognizable sign with a prediction value: {y_pred}" + Style.RESET_ALL)
            #return {"Value": [float(y_pred), class_current]}

        class_list.append(class_current)
    return {'pred': class_list}

@app.get("/")
def root():
    return {'greeting': 'Hello'}
