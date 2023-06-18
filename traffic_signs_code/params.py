import os

IMG_path =  os.path.join(os.getcwd(),'raw_data','Train')
IMG_test_path = os.path.join(os.getcwd(),'raw_data','test_images')
MODELS_path = os.path.join(os.getcwd(),'raw_data','models')
Yolo_Model_path = os.path.join(os.getcwd(),'raw_data','models', 'yolo_v2.pt')
Crop_path= os.path.join(os.getcwd(),'raw_data','Train', 'crop_images')

IMG_WIDTH_BASE = 100
IMG_HEIGHT_BASE = 100
IMG_WIDTH_VGG= 224
IMG_HEIGHT_VGG = 224
