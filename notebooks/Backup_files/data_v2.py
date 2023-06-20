import numpy as np
import os, cv2
from sklearn.utils import shuffle
import splitfolders
from traffic_signs_code.params import *

def create_dataset(data_path):
    splitfolders.ratio(IMG_path, output= Split_path, seed=1337, ratio=(0.8,0.0,0.2))
    os.remove(Split_path + '/val')
    img_data_array=[]
    for label in labels:
        image_path= os.path.join(data_path, label)
        class_num = labels.index(label)
        for img in os.listdir(image_path):
            image= cv2.imread(os.path.join(image_path, img), cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT_VGG_, IMG_WIDTH_VGG_))
            img_data_array.append([image, class_num])
    return np.array(img_data_array)
