FROM tensorflow/tensorflow:2.10.0
COPY traffic_signs_code /traffic_signs_code
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn traffic_signs_code.api.api:app --host 0.0.0.0 --port 8888
