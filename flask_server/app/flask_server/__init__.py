import os
import json
import certifi
from flask import Flask
from elasticsearch import Elasticsearch

from .shared import db, migrate, csrf, ma, jwt, cors
from .email_utils import mail
from .models import Site, Device
from .users.models import User
from .routes import bp_main
from .users.api import bp_user


def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('config.py')
    app.secret_key = os.urandom(16)
    initialize_extensions(app)
    init_es_client(app)
    app.register_blueprint(bp_main)
    app.register_blueprint(bp_user)
    return app

def initialize_extensions(app):
    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    ma.init_app(app)
    jwt.init_app(app)
    cors.init_app(app)
    mail.init_app(app)

def init_es_client(app):
    app.es_client = Elasticsearch(
        hosts=[app.config['ES_HOST']],
        port=app.config['ES_PORT'],
        http_auth=(app.config['ES_USER'],app.config['ES_PWD']),
        use_ssl=True,
        ca_certs=certifi.where(),
        max_retries=0,
        timeout=30
    )
