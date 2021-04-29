from flask import current_app
from flask_server.shared import db, jwt, ma
from werkzeug.security import generate_password_hash
from marshmallow import fields, validate, pre_load


class User(db.Model):  
    __tablename__ = 'nsb_user'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    confirmed_on = db.Column(db.DateTime, nullable=True)

    def __init__(self, username, email, password, is_admin=False):
        self.username = username
        self.email = email
        self.password = generate_password_hash(password, method='sha256')
        self.is_admin = is_admin

    def save(self):
        db.session.add(self)
        db.session.commit()

    def to_dict(self):
        return dict(username=self.username, email=self.email)

    def __repr__(self):
        return self.username


@jwt.user_loader_callback_loader
def user_loader_callback(identity):
    user = User.query.filter(User.username==identity).first()
    if not user:
        return None
    return user


class UserSchema(ma.Schema):
    id = fields.Int(dump_only=True)
    username = fields.Str(
        required=True,
        validate=[validate.Length(min=4, max=20)],
    )
    email = fields.Str(
        required=True,
        validate=validate.Email(error='Not a valid email address'),
    )
    password = fields.Str(
        required=True,
        validate=[validate.Length(min=6, max=36)],
        load_only=True,
    )

    # Clean up data
    @pre_load
    def process_input(self, data, **kwargs):
        data['email'] = data['email'].lower().strip()
        return data

    # We add a post_dump hook to add an envelope to responses
#    @post_dump(pass_many=True)
#    def wrap(self, data, many):
#        key = 'users' if many else 'user'
#        return {
#            key: data,
#        }


user_schema = UserSchema()
