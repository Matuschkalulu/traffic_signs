import os

IMG_PATH =  os.path.join(os.getcwd(),'raw_data','Train')
SPLIT_PATH =  os.path.join(os.getcwd(),'raw_data','split_data')
SPLIT_TRAIN_PATH=  os.path.join(os.getcwd(),'raw_data','split_data', 'train')
SPLIT_TEST_PATH =  os.path.join(os.getcwd(),'raw_data','split_data', 'test')

IMG_TEST_PATH = os.path.join(os.getcwd(),'raw_data','test_images')
MODELS_PATH = os.path.join(os.getcwd(),'raw_data','models')
CROP_PATH= os.path.join(os.getcwd(),'raw_data', 'crop_images')
LABELS = ['readable', 'unreadable']

IMG_WIDTH_VGG= 224
IMG_HEIGHT_VGG = 224
IMG_WIDTH_VGG_= 160
IMG_HEIGHT_VGG_ = 160

LOCAL_MODEL_PATH= os.path.join(os.getcwd(),'raw_data','models', 'final_model')
YOLO_MODEL_PATH= os.path.join(os.getcwd(),'raw_data','models')


VGG_Model_path = os.path.join(os.getcwd(),'raw_data','models', 'model.h5')
Test_Image_path= os.path.join(os.getcwd(),'raw_data','test_images', '00007.jpg')
Street_Image_path= os.path.join(os.getcwd(),'raw_data','street_images', '00001.jpg')
Test_video_path= os.path.join(os.getcwd(),'raw_data','test_videos', 'video_1.mp4')

OUTPUT_PATH= os.path.join(os.getcwd(), 'raw_data', 'output_dir')
INPUT_PATH= os.path.join(os.getcwd(), 'raw_data', 'input_dir')
