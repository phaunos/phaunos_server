FROM tiangolo/uwsgi-nginx-flask:python3.8

ENV NGINX_MAX_UPLOAD 20m

ENV STATIC_URL /static
ENV STATIC_PATH /app/flask_server/static

RUN apt-get update && apt-get install -y libsndfile1 ffmpeg

COPY requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

ENV FLASK_APP flask_server

