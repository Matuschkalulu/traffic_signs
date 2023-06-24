# Classification of Recognizable and Unrecognizable Traffic Signs

<img width="400" height="300" alt="unrecognizable_traffic_sign" src="https://advancelocal-adapter-image-uploads.s3.amazonaws.com/image.silive.com/home/silive-media/width2048/img/seen/photo/2017/08/18/23257976-standard.jpg">

## About
The idea was developed during a short car ride. During that ride
it appeared that quite a lot traffic signs were unrecognizable, which let to unvertanties about which max speed is allowed or who has priority at the crossing?
Comunities are responsible to access free view at the traffic signs to assure every driver's safety.

To attack this problem: This project aims to make a first step to improve that risk.

## Steps
The project follows different goals:

1. Gather a Dataset for unrecognizable traffic signs via web scraping.
2. Crope the images with Yolo model and clean the data.
3. Create a binary classification model via CNN for unrecognizable and recognizable traffic signs.
4. Design an app for detecting and classifying recognizable and un recognizable traffic sugns in images and videos.

## Data Preparation
In this project we didn't have unrecognizable images at hand. Therefore, we had to scrap every website which includes unrecognizable traffic signs. Since the unrecognizable signs are quite a few, we got mismatches at each scraping and therefore had to clean the data and separate the classes. Then, we trained a Yolo_v8 model on traffic signs to use it afterwards for croping.

## Modeling
We tried different CNN architectures. We began with a base model and got a base score. Then, we moved to transfer learning. We started with VGG16	which was a bit underfitting the task. We then used a more complex residual net structure; Resnet50 which performed much better. Since we didn't have a large dataset, we applied different techniques of data augmentation to both the samples and the model layers. We achieved a test score of 99% on unseen test dataset with Resnet50.

file:///home/maly/Downloads/history_resnet.png
