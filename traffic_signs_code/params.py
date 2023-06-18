import os

IMG_path =  os.path.join(os.getcwd(),'raw_data','Train')
Split_path =  os.path.join(os.getcwd(),'raw_data','split_data')
Train_path =  os.path.join(os.getcwd(),'raw_data','split_data', 'train')
Test_path =  os.path.join(os.getcwd(),'raw_data','split_data', 'test')

IMG_test_path = os.path.join(os.getcwd(),'raw_data','test_images')
MODELS_path = os.path.join(os.getcwd(),'raw_data','models')
Yolo_Model_path = os.path.join(os.getcwd(),'raw_data','models', 'yolo_v2.pt')
Crop_path= os.path.join(os.getcwd(),'raw_data', 'crop_images')

labels = ['readable', 'unreadable']
IMG_WIDTH_BASE = 100
IMG_HEIGHT_BASE = 100
IMG_WIDTH_VGG= 224
IMG_HEIGHT_VGG = 224
IMG_WIDTH_VGG_= 160
IMG_HEIGHT_VGG_ = 160
