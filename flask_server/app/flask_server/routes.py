import os
import time
import re
from datetime import datetime, timedelta
from dateutil import tz
import audioread
from flask import (render_template, request, current_app,
                   Blueprint, send_from_directory, session,
                   jsonify, url_for)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from flask_jwt_extended import jwt_optional, verify_jwt_in_request, get_jwt_identity
from celery.result import AsyncResult
from .shared import csrf
from .models import Device
from .celery_app import celery


bp_main = Blueprint('bp_main',  __name__)


def build_error_response(msg, status_code):
    resp = jsonify(error=msg)
    resp.status_code = status_code
    return resp


@bp_main.errorhandler(RequestEntityTooLarge)
def error413(e):
    return build_error_response(
        f"The file must be smaller than {current_app.config['MAX_CONTENT_LENGTH'] / 1024**2} MB",
        413)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@bp_main.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'],
                               filename)

@bp_main.route('/', methods=['GET'])
def index():
    devices = Device.query.order_by(Device.name).all()
    try:
        verify_jwt_in_request()
        current_user = get_jwt_identity()
        return render_template(
            'index.html',
            auth_container=render_template('auth/logout.html', current_user=current_user),
            current_user=current_user,
            devices=devices)
    except Exception as e:
        return render_template(
            'index.html',
            auth_container=render_template('auth/login.html'),
            devices=devices)


@bp_main.route('/_upload', methods=['POST'])
def _upload():

    ###########################################
    # Convert file to mono, 16 bits, 22050 Hz #
    ###########################################

    # check if the post request has the file part
    if 'file' not in request.files:
        return build_error_response("No selected file.", 404)

    file = request.files['file']
    device_id = int(request.form.get('device_id'))
    timestamp_str = request.form.get('timestamp')

    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return build_error_response("No selected file", 404)
    if not (file and allowed_file(file.filename)):
        return build_error_response("File format not allowed.", 404)

    # check device_id
    if device_id == -1:
        return build_error_response("Please select a device.", 404)
    device = Device.query.get(device_id)

    # check timestamp format
    try:
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
        timestamp = timestamp.replace(tzinfo=tz.tzutc())
        # set timestamp to Paris time
        # TODO set according to device position
        timestamp = timestamp.astimezone(tz.gettz('Europe/Paris'))
    except ValueError:
        return build_error_response("Please make sure the timestamp matches the template.", 404)


    filename = secure_filename(file.filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

    if os.path.exists(filepath):
        os.remove(filepath)
    file.save(filepath)
    
    # make sure data from same device and time range does not already exist
    af = audioread.audio_open(filepath)
    duration = af.duration
    af.close()

    count = current_app.es_client.count(
        index=current_app.config['ES_INDEX'],
        body={"query":
              {"bool": {"must": [
                  {"bool": {"must": [
                      {"range": {"audiofile.startTime" : {"lte": f"{(timestamp + timedelta(seconds=duration)).isoformat()}"}}},
                      {"range": {"audiofile.endTime": {"gte": f"{timestamp.isoformat()}"}}}]}
                   },
                  {"match": {"device.name": f"{device.name}"}}
              ]}}})

    if count['count']:
        os.remove(filepath)
        return build_error_response("Some detections for this device and in this time range already exists.", 404)

    task = celery.send_task(
        'tasks.process',
        args=[
            filename,
            timestamp.isoformat(),
            current_app.config['CHUNK_DURATION'],
            device.name,
            device.lat,
            device.lon,
            current_app.config['PROJECT_NAME']
        ])

    return jsonify({}), 202, {'location': url_for('bp_main.taskstatus',
                                              task_id=task.id)}

@bp_main.route('/status/<task_id>')
def taskstatus(task_id):
    task = AsyncResult(task_id)

    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'status': task.info['msg']
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'status': task.info['msg']
        }
    elif task.state == 'FAILURE':
        response = {
            'state': task.state,
            'status': [str(task.info)]
        }
    else:
        response = {
            'state': task.state,
            'status': 'Unknown state'
        }
    return jsonify(response)
