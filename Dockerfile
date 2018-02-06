FROM python:2

RUN mkdir ./home/TensorFlow
WORKDIR ./home/TensorFlow
RUN mkdir ./src
COPY ./src ./src

CMD ["python", "./src/HelloWorld.py"]
