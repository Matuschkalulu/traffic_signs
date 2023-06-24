from colorama import Fore, Style
from traffic_signs_code.params import *
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import splitfolders
import shutil

def create_dataset(model_selected = 'Base'):
    '''
    This functions aims to create the dataset containing the images and their classification
    The binary classification only considers whether the sign is reconizable or not
    =========
    Input
    As input to this function some has to select a model, to decide whether the image has to be normalized or not
    The default input is "Base"
    Other inputs: "VGG"

    =========
    Return:
    The function returns an array containing all the images and the class names
        image_data_array : will be used as the feature parametes for the model
        class_name: will be used as the bases for the target
    '''
    img_folder = IMG_PATH
    img_data_array = []
    class_name = []
    IMG_HEIGHT = IMG_HEIGHT_VGG_
    IMG_WIDTH = IMG_WIDTH_VGG_

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
             if file.endswith('.png'):
                image_path= os.path.join(img_folder, dir1,  file)
                image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
                image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
                image=np.array(image)
                image = image.astype('float32')
                img_data_array.append(image)
                class_name.append(dir1)
    print("\N{white heavy check mark}" +" Data was sucessfully created")
    return img_data_array, class_name

def create_dataset_with_split(data_path):
    '''
    This functions aims to create the dataset containing the images and their classification
    The binary classification only considers whether the sign is reconizable or not
    =========
    Input:
    Data_path of either the test or train data created by the splitfolders() function

    =========
    Return: -> np.array
    Returns a numpy array with the images an their labels
    '''
    print(Fore.BLUE + "\nStart of splitting data into train and test..." + Style.RESET_ALL)
    splitfolders.ratio(IMG_PATH, output= SPLIT_PATH, seed=1337, ratio=(0.8,0.0,0.2))
    shutil.rmtree(SPLIT_PATH + '/val')
    img_data_array=[]
    for label in LABELS:
        image_path= os.path.join(data_path, label)
        class_num = LABELS.index(label)
        for img in os.listdir(image_path):
            if img.endswith('.png') or  img.endswith('.jpg'):
                image= cv2.imread(os.path.join(image_path, img), cv2.COLOR_BGR2RGB)
                image=cv2.resize(image, (IMG_HEIGHT_VGG_, IMG_WIDTH_VGG_))
                img_data_array.append([image, class_num])

    print('✅ Data was sucessfully created and saved into local disk')
    return np.array(img_data_array)

# Visualize the some retrieved data
def visualize_data(data):
    l = []
    for i in data:
        if(i[1] == 0):
            l.append("readable")
        else:
            l.append("unreadable")

    # Investigate the Classes
    unique, count= np.unique(l, return_counts=True)
    unique_count=dict(zip(unique, count))

    # Visualize one sample of each class
    plt.figure(figsize = (5,5))
    plt.imshow(data[1][0])
    plt.title(LABELS[data[0][1]])
    plt.imshow(data[-1][0])
    plt.title(LABELS[data[-1][1]])

# Function to crop data
def crop_images(model_name = 'yolo_v2.pt'):
    """
    Crops out the traffic signs via YOLO detection to push it to the classification model:
    =======
    Input :
    model_name : The model hast to be called by its name to make the prediction. A default is set.
    ======
    Return: None -> Funcitons save the croped image in a directory
    """
    print(Fore.BLUE + "\nStart of croping images..." + Style.RESET_ALL)
    custom_model = YOLO(os.path.join(YOLO_MODEL_PATH, 'yolo_v2.pt'))
    img_path= IMG_PATH
    img_list=[]
    for current_dir, dirs, files in os.walk(img_path):
        for dir_ in dirs:
            if dir_!='.ipynb_checkpoints':
                for f in os.listdir(img_path + dir_):
                    #print(f)
                    img = cv2.imread(os.path.join(img_path, dir_, f), cv2.IMREAD_COLOR)
                    img_list.append(img)
                    results = custom_model(img)
                    for n, box in enumerate(results[0].boxes.xywhn):
                        h, w = img.shape[:2]
                        x1, y1, x2, y2 = box.numpy()
                        x_center, y_center = int(float(x1) * w), int(float(y1) * h)
                        box_width, box_height = int(float(x2) * w), int(float(y2) * h)
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))
                        crop_img= img[y_min:y_min+int(box_height), x_min:x_min+int(box_width)]
                        #plt.imshow(crop_img)
                        #plt.show();
                        save_path= CROP_PATH
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        cv2.imwrite(save_path + f"{f}", crop_img)
                        print('✅ Images are successfully croped and saved to local disk')
