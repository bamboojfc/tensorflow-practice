FROM python:2

RUN mkdir ./home/TensorFlow
WORKDIR ./home/TensorFlow
RUN mkdir ./src
COPY ./src ./src

RUN pip install tensorflow
CMD ["python", "./src/CreateGraphExercise.py"]
