import os

IMG_unrecognizable_path = os.path.join(os.getcwd(),'raw_data','Train','0')
IMG_recognizable_path = os.path.join(os.getcwd(),'..','raw_data','Train','1')
IMG_path =  os.path.join(os.getcwd(),'raw_data','Train')

IMG_WIDTH = 100
IMG_HEIGHT = 100
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
