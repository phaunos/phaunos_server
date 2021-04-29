from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_wtf.csrf import CSRFProtect
from flask_jwt_extended import JWTManager
from flask_marshmallow import Marshmallow
from flask_cors import CORS


db = SQLAlchemy()
migrate = Migrate()
csrf = CSRFProtect()
ma = Marshmallow()
jwt = JWTManager()
cors = CORS(resources={r"*": {"origins": "*"}}, expose_headers=['Access-Control-Allow-Origin'], supports_credentials=True)
