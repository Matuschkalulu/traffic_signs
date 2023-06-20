from ultralytics import YOLO
import cv2, os
from traffic_signs_code.params import *

# Function to crop data
def crop_images():
    custom_model = YOLO(Yolo_Model_path)
    img_path= IMG_path
    img_list=[]
    for current_dir, dirs, files in os.walk(img_path):
        for dir_ in dirs:
            if dir_!='.ipynb_checkpoints':
                for f in os.listdir(img_path + dir_):
                    #print(f)
                    img = cv2.imread(os.path.join(img_path, dir_, f), cv2.IMREAD_COLOR)
                    img_list.append(img)
                    results = custom_model(img)
                    for n, box in enumerate(results[0].boxes.xywhn):
                        h, w = img.shape[:2]
                        x1, y1, x2, y2 = box.numpy()
                        x_center, y_center = int(float(x1) * w), int(float(y1) * h)
                        box_width, box_height = int(float(x2) * w), int(float(y2) * h)
                        x_min = int(x_center - (box_width / 2))
                        y_min = int(y_center - (box_height / 2))
                        crop_img= img[y_min:y_min+int(box_height), x_min:x_min+int(box_width)]
                        #plt.imshow(crop_img)
                        #plt.show();
                        save_path= Crop_path
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        cv2.imwrite(save_path + f"{f}", crop_img)
