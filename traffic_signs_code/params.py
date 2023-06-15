import os

IMG_path =  os.path.join(os.getcwd(),'raw_data','Train')

IMG_WIDTH_BASE = 100
IMG_HEIGHT_BASE = 100
IMG_WIDTH_VGG= 224
IMG_HEIGHT_VGG = 224
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
