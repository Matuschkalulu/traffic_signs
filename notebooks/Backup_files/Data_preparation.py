import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os, cv2, glob
import requests
from bs4 import BeautifulSoup as bs
import urllib, urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time, shutil
import random

# Data Visualization
df_train= pd.read_csv('data/train.csv')
df_train['Path']= df_train['Path'].apply( lambda x:x.lower())

fig= plt.figure(figsize=(12,6))
image= [cv2.imread(img) for img in glob.glob('data/train/0/*.png')]
for i in range(0,10):
    fig.add_subplot(2,5,i+1)
    plt.imshow(image[i])
plt.show()

sns.countplot(df_train['ClassId'])

# Web Scraping
def scrap_unrecognized_images(url: str, search_word: str, keyword: str) -> None:
    driver= webdriver.Chrome(ChromeDriverManager().install())
    driver.get(url)
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
    time.sleep(5)
    imgResults = driver.find_elements(By.XPATH,f"//img[contains(@class,'{search_word}')]")

    src = []
    for img in imgResults:
        src.append(img.get_attribute('src'))
    save_path=f'data/stop_unrecognized/{keyword}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i,v in enumerate(src):
        if v != None:
            urllib.request.urlretrieve(str(src[i]), os.path.join(save_path, "{}.png".format(i)))
            print(f'scraping url number {i} out of {len(src)}')
        else:
            print(f'found {src[i]} type at {i}')
    return None

# Random Selection of Images from each Subfolder in the Recognizable Dataset
save_path=f'data/train_all'
train_folders= 'data/train/'
img_list=[]
images_count=[]
for folder in os.listdir(train_folders):
    dirpath= os.path.join(train_folders, folder)
    for img in glob.glob(dirpath + '/' + '*.png'):
        img_list.append(img)
    images= random.sample(img_list,7)
    img_list=[]
    img_src= ','.join([x for x in images])
    image= [image.split('/')[-1] for image in images]
    image= ','.join(image)
    for i in image.split(','):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        shutil.copy2(os.path.join(train_folders, folder, i), os.path.join(save_path, i))
