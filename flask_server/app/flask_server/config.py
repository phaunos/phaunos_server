import os


PROJECT_NAME = os.environ['PROJECT_NAME'] # TODO Get from DB according to user

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
CHUNK_DURATION = 30 # the audio file will be split in chunks of CHUNK_DURATION seconds
MAX_CONTENT_LENGTH = 20 * 1024 * 1024 # 20 MB

CELERY_BROKER_URL = os.environ['CELERY_BROKER_URL']
CELERY_RESULT_BACKEND = os.environ['CELERY_RESULT_BACKEND']

SQLALCHEMY_DATABASE_URI = 'sqlite:////db/test.db'
SQLALCHEMY_TRACK_MODIFICATIONS = False

CONFIRMATION_TOKEN_EXPIRATION = int(os.environ['CONFIRMATION_TOKEN_EXPIRATION'])

MAIL_SERVER = os.environ['MAIL_SERVER']
MAIL_PORT = int(os.environ['MAIL_PORT'])
MAIL_USE_TLS = int(os.environ['MAIL_USE_TLS']) == 1
MAIL_USE_SSL = int(os.environ['MAIL_USE_SSL']) == 1
MAIL_DEFAULT_SENDER = os.environ['MAIL_DEFAULT_SENDER']
MAIL_USERNAME = os.environ['MAIL_USERNAME']
MAIL_PASSWORD = os.environ['MAIL_PASSWORD']

SECURITY_PASSWORD_SALT = os.environ['SECURITY_PASSWORD_SALT']

JWT_TOKEN_LOCATION = ('headers', 'cookies')
JWT_SECRET_KEY = os.environ['JWT_SECRET_KEY']
JWT_ACCESS_TOKEN_EXPIRES = int(os.environ['JWT_ACCESS_TOKEN_EXPIRES'])
JWT_REFRESH_TOKEN_EXPIRES = int(os.environ['JWT_REFRESH_TOKEN_EXPIRES'])
JWT_COOKIE_CSRF_PROTECT = int(os.environ['JWT_COOKIE_CSRF_PROTECT']) == 1
JWT_COOKIE_SECURE = int(os.environ['JWT_COOKIE_SECURE']) == 1

ES_HOST = os.environ['ES_HOST']
ES_INDEX = os.environ['ES_INDEX']
ES_PORT = os.environ['ES_PORT']
ES_USER = os.environ['ES_USER']
ES_PWD = os.environ['ES_PWD']

