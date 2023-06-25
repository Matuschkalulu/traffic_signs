# Classification of Recognizable and Unrecognizable Traffic Signs

<img width="400" height="250" alt="unrecognizable_traffic_sign" src="https://advancelocal-adapter-image-uploads.s3.amazonaws.com/image.silive.com/home/silive-media/width2048/img/seen/photo/2017/08/18/23257976-standard.jpg">

## About
The idea was developed during a short car ride. During that ride
it appeared that quite a lot traffic signs were unrecognizable, which let to unvertanties about which max speed is allowed or who has priority at the crossing?
Comunities are responsible to access free view at the traffic signs to assure every driver's safety. We decided to attack this problem and take the first step to improve that risk.

## Steps
The project follows different goals:

1. Gather a Dataset for unrecognizable traffic signs via web scraping.
2. Crope the images with Yolo model and clean the data.
3. Create a binary classification model via CNN for unrecognizable and recognizable traffic signs.
4. Design an app for detecting and classifying recognizable and un recognizable traffic sugns in images and videos.

## Data Preparation
In this project we didn't have unrecognizable images at hand. Therefore, we had to scrap every website which includes unrecognizable traffic signs. Since the unrecognizable signs are quite a few, we got mismatches at each scraping and therefore had to clean the data and separate the classes. Then, we trained a Yolo_v8 model on traffic signs to use it afterwards for croping.
The data tree was as shown.

![Screenshot from 2023-06-22 09-18-25](https://github.com/Matuschkalulu/traffic_signs/assets/107108097/948179a1-86c9-4e7d-9b7b-da374e49d6c1)


## Modeling
We tried different CNN architectures. We began with a base model and got a base score. Then, we moved to transfer learning. We started with VGG16	which was a bit underfitting the task. We then used a more complex residual net structure; Resnet50 which performed much better. Since we didn't have a large dataset, we applied different techniques of data augmentation to both the samples and the model layers. We achieved a test score of 99% on unseen test dataset with Resnet50.

## Results
![history_resnet](https://github.com/Matuschkalulu/traffic_signs/assets/107108097/8a5b076c-8bbc-4abc-bff6-041bbd7fc21c)

![classification_report](https://github.com/Matuschkalulu/traffic_signs/assets/107108097/d6574314-7fee-4c6f-8e4b-7b3b2a590907)

## Local Run
1. Clone the repo.
2. run "pip install -e ." to install all necessary packages of the project on your local disk.
3. For model training and evaluation, run the main.py file which collects all necessary methods.
4. For testing the model for images and videos, save an image or video to the ./raw_data/input_dir path and run the ./traffic_signs_code/video_detection/video_detection.py file.

## Run on the Cloud
You can run the api file to test the model using the prebuilt docker image on the cloud. To do that, run "make run_api". This will open an api local host, add /docs at the end of the url to open the FastAPI and you can try an image or video from there.

## Run Streamlit Application
Instead of cloning the repo, you can run the app directly with https://trafficsignsdet.streamlit.app/.


![f6f4c7d9d53ee5bcedb288601eccecb56d04d8469618d93a961fc803](https://github.com/Matuschkalulu/traffic_signs/assets/107108097/e51c676e-e557-4ae7-890a-0f9c0388df59)


![img_readme2](https://github.com/Matuschkalulu/traffic_signs/assets/107108097/1bb6de38-e369-47a4-85fd-0c3d607c69ab)


Feel free to try it and happy driving!
