#Comment for merge
FROM tensorflow/tensorflow:2.10.0
COPY traffic_signs_code /traffic_signs_code
COPY models/first_model /models/first_model
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install python-multipart
CMD uvicorn traffic_signs_code.api.api:app --host 0.0.0.0 --port $PORT
