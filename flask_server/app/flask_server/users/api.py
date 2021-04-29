import datetime
from flask import (
    request,
    Blueprint,
    jsonify,
    make_response,
    url_for,
    render_template,
    redirect,
    current_app,
)
from flask_jwt_extended import (
    jwt_refresh_token_required,
    get_jwt_identity,
    create_access_token,
    create_refresh_token,
    set_access_cookies,
    set_refresh_cookies,
    unset_jwt_cookies
)
from flask_server.shared import db, csrf
from flask_server.users.models import User, user_schema
import re
from marshmallow import ValidationError
from werkzeug.security import check_password_hash
from ..email_utils import send_confirmation_email, generate_confirmation_token, confirm_token
from sqlalchemy import exc


bp_user = Blueprint('bp_user',  __name__)


@bp_user.route('/api/user/create_user', methods=('POST',))
@csrf.exempt
def create_user():

    data = request.get_json()
    web = request.args.get('web', 0, type=int)

    try:

        # Validate schema
        data = user_schema.load(data)

        if User.query.filter(User.username == data['username']).first():
            return jsonify({'username': 'This username is already in the database'}), 409
        # Create unconfirmed user if email is not already registered
        if User.query.filter(User.email == data['email']).first():
            return jsonify({'email': 'This email address is already in the database'}), 409
        user = User(data['username'],
            email=data['email'],
            password=data['password'],
        )
        user.save()

        # Send confirmation email
        token = generate_confirmation_token(user.email)
        send_confirmation_email(user.email, token)

        return jsonify({'msg': f'A confirmation email has been sent to {user.email}.'}), 201

    except ValidationError as err:
        return jsonify(err.messages), 400
    
    except exc.SQLAlchemyError:
        return jsonify({'msg': 'SQLAlchemy error.'}), 500


@bp_user.route('/api/user/request_confirmation_email', methods=('POST',))
def request_confirmation_email():

    data = request.get_json()

    user = User.query.filter(User.email == data['email']).first()
    
    if not user:
        return jsonify({'email': f'{user.email} not found'}), 400

    if user.confirmed_on:
        return jsonify({'email': f'{user.email} already confirmed'}), 400

    token = generate_confirmation_token(user.email)
    send_confirmation_email(user.email, token)
    return jsonify({'msg': f'A confirmation email has been sent to {user.email}'}), 200


@bp_user.route('/api/user/confirm/<token>')
def confirm_email(token):

    try:
        email = confirm_token(token, current_app.config['CONFIRMATION_TOKEN_EXPIRATION'])
    except:
        return jsonify({'msg':'The confirmation link is invalid or has expired.'}), 400

    user = User.query.filter_by(email=email).first()
    if user.confirmed_on:
        return jsonify({'msg':f'{user.email} already confirmed'}), 400
    else:
        user.confirmed_on = datetime.datetime.now()
        db.session.add(user)
        db.session.commit()
        return jsonify({'msg':'You have confirmed your account. Thank you!'}), 200
    return redirect(url_for('phaunos_api.home'))


@bp_user.route('/api/user/login', methods=('POST',))
@csrf.exempt
def login():

    data = request.values

    user = User.query.filter(User.username==data['username']).first()

    if not user:
        return jsonify({'msg': 'User {} doesn\'t exist'.format(data['username'])}), 400

    if not user.confirmed_on:
        return jsonify({'msg': f'{user.email} not confirmed'}), 400

    if check_password_hash(user.password, data['password']):
        access_token = create_access_token(identity = data['username'], fresh=True)
        refresh_token = create_refresh_token(identity = data['username'])
        resp = jsonify({
            'html': render_template('auth/logout.html', current_user=user.username)})
        set_access_cookies(resp, access_token)
        set_refresh_cookies(resp, refresh_token)
        return resp, 200

    else:
        return jsonify({'msg': 'Invalid password.'}), 401


@bp_user.route('/api/user/logout', methods=['GET'])
def logout():
    resp = jsonify({
        'html': render_template('auth/login.html')})
    unset_jwt_cookies(resp)
    return resp, 200


# Refresh token endpoint. This will generate a new access token from
# the refresh token, but will mark that access token as non-fresh,
# as we do not actually verify a password in this endpoint.
@bp_user.route('/api/token/refresh', methods=['POST'])
@jwt_refresh_token_required
def refresh_token():
    current_user = get_jwt_identity()
    new_token = create_access_token(identity=current_user, fresh=False)
    ret = {'access_token': new_token}
    return jsonify(ret), 200

