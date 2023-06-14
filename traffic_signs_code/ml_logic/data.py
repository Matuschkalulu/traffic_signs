from traffic_signs_code.params import *
import os
import cv2
import numpy as np

def create_dataset(model_selected = 'Base'):
    '''
    This functions aims to create the dataset containing the images and their classification
    The binary classification only considers whether the sign is reconizable or not
    =========
    Input
    As input to this function some has to select a model, to decide whether the image has to be normalized or not
    The default input is "Base"
    Other inputs: "Vgg"

    =========
    Return:
    The function returns an array containing all the images and the class names
        image_data_array : will be used as the feature parametes for the model
        class_name: will be used as the bases for the target
    '''

    img_folder =IMG_path
    img_data_array=[]
    class_name=[]

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path= os.path.join(img_folder, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            if model_selected ==  'Base':
                image /= 255
            img_data_array.append(image)
            class_name.append(dir1)

    return img_data_array, class_name
