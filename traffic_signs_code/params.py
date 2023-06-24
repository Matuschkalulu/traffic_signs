import os

IMG_PATH =  os.path.join(os.getcwd(),'raw_data','Train')
SPLIT_PATH =  os.path.join(os.getcwd(),'raw_data','split_data')
SPLIT_TRAIN_PATH=  os.path.join(os.getcwd(),'raw_data','split_data', 'train')
SPLIT_TEST_PATH =  os.path.join(os.getcwd(),'raw_data','split_data', 'test')

CROP_PATH= os.path.join(os.getcwd(),'raw_data', 'crop_images')
LABELS = ['readable', 'unreadable']

IMG_WIDTH_VGG_= 160
IMG_HEIGHT_VGG_ = 160

LOCAL_MODEL_PATH= os.path.join(os.getcwd(),'raw_data','models', 'first_model')
YOLO_MODEL_PATH= os.path.join(os.getcwd(),'raw_data','models')

OUTPUT_PATH= os.path.join(os.getcwd(), 'raw_data', 'output_dir')
INPUT_PATH= os.path.join(os.getcwd(), 'raw_data', 'input_dir')
