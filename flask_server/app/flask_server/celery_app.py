from celery import Celery

from flask_server import config


celery = Celery(
    'tasks',
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND)
