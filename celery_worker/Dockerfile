FROM python:3.8-buster

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

#COPY . /worker
#WORKDIR worker

ENV PYTHONPATH "${PYTHONPATH}:/worker:/nsb_aad:/phaunos_ml"
ENV NUMBA_CACHE_DIR "/tmp/cache1"
