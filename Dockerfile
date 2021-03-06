FROM ubuntu:xenial

RUN mkdir ./home/TensorFlow
WORKDIR ./home/TensorFlow
RUN mkdir ./src
COPY ./src ./src

RUN apt-get update && apt-get install -y \
    python2.7 \
    python-pip
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["/usr/bin/python2.7"]
CMD ["python", "./src/test.py"]
