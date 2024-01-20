FROM tensorflow/tensorflow:latest
WORKDIR /opt
RUN pip3 install keras-ocr matplotlib
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
CMD [ "python", "/opt/lab2.py" ] 