import pandas as pd
import numpy as np
import tensorflow as tf
import colorama
from colorama import Fore, Style
import cv2, os, time
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
#from traffic_signs_code.video_detection.video_detection import process_video
from traffic_signs_code.params import *
import matplotlib.pyplot as plt
from PIL import Image
import io
import ipdb

custom_model = YOLO(os.path.join(LOCAL_MODEL_PATH, 'yolo_v2.pt'))

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.state.model= load_model()

@app.post("/ImagePrediction/")
async def create_prediction(file: UploadFile = File(...)):
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    pred_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    custom_model = YOLO(os.path.join(LOCAL_MODEL_PATH, 'best_v2.pt'))
    results = custom_model(pred_image, device='cpu')

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
    return {'Value': class_list, 'Actual Prediction Value': [float(i) for i in pred_list]}

@app.post("/VideoPrediction/")
async def video_prediction(file: UploadFile = File(...)):
    custom_model = YOLO(os.path.join(LOCAL_MODEL_PATH, 'yolo_v2.pt'))
    app.state.model= load_model()
    app.state.model.compile(loss='binary_crossentropy',
                    optimizer ='adam',
                    metrics=['accuracy'])

    with open('test_video.mp4', "wb") as buffer:
        buffer.write(await file.read())
    current_directory = os.getcwd()
    main_video_path = os.path.join(current_directory, 'test_video.mp4')

    video= cv2.VideoCapture(main_video_path)

    writer = None
    h, w = None, None
    f = 0
    t = 0

    # Catching frames in the loop

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if w is None or h is None:
            h, w = frame.shape[:2]
        start= time.time()
        results= custom_model(frame)
        end= time.time()
        f += 1
        t += end - start

        # Spent time for current frame
        print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

        crop_list=[]
        cord_list=[]
        for n, box in enumerate(results[0].boxes.xywhn):
            x1, y1, x2, y2 = box.numpy()
            x_center, y_center = int(float(x1) * w), int(float(y1) * h)
            box_width, box_height = int(float(x2) * w), int(float(y2) * h)
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))
            crop_frame= frame[y_min:y_min+int(box_height), x_min:x_min+int(box_width)]
            crop_list.append(crop_frame)
            cord_list.append([x_min, y_min, box_width, box_height])

        class_list=[]
        score_list=[]
        for arr in crop_list:
            image= cv2.resize(arr, (IMG_WIDTH_VGG_, IMG_HEIGHT_VGG_), interpolation = cv2.INTER_AREA)
            image= np.expand_dims(image, axis=0)
            img_preprocessed = preprocess_input(image)
            score= app.state.model.predict(img_preprocessed)[0][0]
            score_list.append(score)
            if score < 0.6:
                class_current= 'READABLE'
            else:
                class_current= 'UNREADABLE'
            class_list.append(class_current)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(os.path.join(os.getcwd(), 'output_video.mp4'), fourcc, 25,
                                     (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)

    print('Total number of frames', f)
    print('Total amount of time {:.5f} seconds'.format(t))
    print('FPS:', round((f / t), 1))
    video.release()
    writer.release()
    return {'pred': [float(i) for i in score_list]}

    #score_list, class_list= process_video(main_video_path, app.state.model)
    #return {'pred': zip(score_list, class_list)}

@app.get("/")
def root():
    return {'greeting': 'Hello'}
