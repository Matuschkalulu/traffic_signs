from fastapi import FastAPI, Response, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from traffic_signs_code.ml_logic.miscfunc import load_model
from traffic_signs_code.params import *
from traffic_signs_code.video_detection.video_detection import *

custom_model = YOLO(os.path.join(YOLO_MODEL_PATH, 'best_v2.pt'))
app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
app.state.model= load_model()
model= app.state.model

@app.post("/ImagePrediction/")
async def create_prediction(file: UploadFile = File(...)):
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    pred_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    crop_list, cord_list= process_file(pred_image)
    pred_list, class_list= pred(crop_list, model)
    visualize_pred(pred_image, cord_list, class_list, plot=False)

    headers = {
        "Content-Disposition": "attachment; filename=output_image.png",
        "Content-Type": "image/png",
    }
    output_img= os.path.join(OUTPUT_PATH,'output_image.png')
    with open(output_img, "rb") as f:
        file_content = f.read()
    response = Response(content=file_content,headers=headers)
    return response

@app.post("/VideoPrediction/")
async def video_prediction(file: UploadFile = File(...)):
    input_video= os.path.join(INPUT_PATH,'test_video.mp4')
    with open(input_video, "wb") as buffer:
        buffer.write(await file.read())

    current_directory = os.getcwd()
    main_video_path = os.path.join(INPUT_PATH, 'test_video.mp4')
    process_video(main_video_path, model)
    headers = {
        "Content-Disposition": "attachment; filename=output_video.mp4",
        "Content-Type": "video/mp4",
    }
    labeled_video_path = os.path.join(OUTPUT_PATH,'output_video.mp4')
    with open(labeled_video_path, "rb") as f:
        file_content = f.read()
    response = Response(content=file_content,headers=headers)
    return response

@app.get("/")
def root():
    return {'greeting': 'Hello'}
