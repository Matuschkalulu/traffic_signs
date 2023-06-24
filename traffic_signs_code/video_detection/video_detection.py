import colorama
from colorama import Fore, Style
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import cv2, os, time
from tensorflow.keras.applications.resnet50 import preprocess_input
from traffic_signs_code.params import *
from traffic_signs_code.ml_logic.miscfunc import load_model


custom_model = YOLO(os.path.join(YOLO_MODEL_PATH, 'yolo_v2.pt'))
model= load_model()

# Detect and Recognize images
def process_file(file):
    results = custom_model(file, device='cpu')
    crop_list=[]
    cord_list=[]
    for n, box in enumerate(results[0].boxes.xywhn):
        x1, y1, x2, y2 = box.numpy()
        h, w = file.shape[:2]
        x_center, y_center = int(float(x1) * w), int(float(y1) * h)
        box_width, box_height = int(float(x2) * w), int(float(y2) * h)
        x_min = int(x_center - (box_width / 2))
        y_min = int(y_center - (box_height / 2))
        crop_img= file[y_min:y_min+int(box_height), x_min:x_min+int(box_width)]
        crop_list.append(crop_img)
        cord_list.append([x_min, y_min, box_width, box_height])
    return crop_list, cord_list

# Prediction
def pred(crop_list, model):
    class_list=[]
    score_list=[]
    for arr in crop_list:
        image= cv2.resize(arr, (IMG_WIDTH_VGG_, IMG_HEIGHT_VGG_), interpolation = cv2.INTER_AREA)
        image= np.expand_dims(image, axis=0)
        img_preprocessed = preprocess_input(image)
        score= model.predict(img_preprocessed)[0][0]
        print(score)
        score_list.append(score)
        if score < 0.5:
            class_current= 'READABLE'
            print(Fore.YELLOW + f"\nThis is an recognizable sign with a prediction value: {score}" + Style.RESET_ALL)
        else:
            class_current= 'UNREADABLE'
            print(Fore.GREEN + f"\nThis is an unrecognizable sign with a prediction value: {score}" + Style.RESET_ALL)
        class_list.append(class_current)
    return score_list, class_list

# Visualize the image with detected and recognized signs
def visualize_pred(file, cord_list, class_list, plot=True, save=True):
    for (c,i) in zip(cord_list, class_list):
        x_min, y_min= c[0], c[1]
        box_width, box_height= c[2], c[3]
        cv2.rectangle(file, (x_min, y_min), (x_min + box_width, y_min + box_height), (0 , 255, 0), 2)
        cv2.putText(file, i, (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 2)
    if save:
        cv2.imwrite(os.path.join(OUTPUT_PATH, 'output_image.png'), file)
    if plot:
        # Plotting the test image
        plt.rcParams['figure.figsize'] = (15, 15)
        fig = plt.figure()
        plt.imshow(cv2.cvtColor(file, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Image with Traffic Signs', fontsize=18)
        plt.show()
        # Saving the plot
        fig.savefig(os.path.join(OUTPUT_PATH, 'output_fig.png'))
        plt.close()

def process_video(file, model):
    """
    =======
    Input:
    file: Video file to process
    model_name: which model will be used for the calssification
    """
    video = cv2.VideoCapture(file)
    writer = None
    h, w = None, None
    f = 0
    t = 0
    # Catching frames in the loop
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if w is None or h is None:
            h, w = frame.shape[:2]
        start= time.time()
        crop_list, cord_list= process_file(frame)
        end= time.time()
        f += 1
        t += end - start
        print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))
        score_list, class_list= pred(crop_list, model)
        visualize_pred(frame, cord_list, class_list, plot=False, save=False)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(os.path.join(OUTPUT_PATH,'output_video.mp4'), fourcc, 25,
                                     (frame.shape[1], frame.shape[0]), True)
        writer.write(frame)
    print('Total number of frames', f)
    print('Total amount of time {:.5f} seconds'.format(t))
    print('FPS:', round((f / t), 1))

    # Releasing video reader and writer
    video.release()
    writer.release()
