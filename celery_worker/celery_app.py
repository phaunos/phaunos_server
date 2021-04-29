import os
import json
import subprocess
import time
import uuid
from datetime import datetime, timedelta
import dateutil.parser
import certifi

import requests
from celery import Celery
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import boto3
from botocore.exceptions import ClientError

import numpy as np
from nsb_aad.frame_based_detectors.mario_detector import MarioDetector
from phaunos_ml.utils import feature_utils
from phaunos_ml.utils.audio_utils import load_audio


BIRD_DETECTION_THRESHOLD = 0.5
BIRD_CLASSIFICATION_THRESHOLD = 0.2
AUDIO_SEGMENT_MIN_SIZE = 2

celery = Celery(
    'tasks',
    broker=os.environ['CELERY_BROKER_URL'],
    backend=os.environ['CELERY_RESULT_BACKEND'])

actdet_cfg_file = '/config/actdet_config.json'
featex_cfg_file = '/config/featex_config.json'

activity_detector_cfg = json.load(open(actdet_cfg_file, 'r'))
min_activity_dur = activity_detector_cfg['min_activity_dur']
activity_detector = MarioDetector(activity_detector_cfg)
feature_extractor = feature_utils.MelSpecExtractor.from_config(featex_cfg_file)

label_file = '/config/labels.txt'
target_label_ids_file = '/config/target_label_ids.txt'

labels = [l.strip() for l in open(label_file, 'r').readlines()]
target_label_ids = [int(ind) for ind in open(target_label_ids_file, 'r').readline().strip().split(',')]
target_labels = [labels[ind] for ind in target_label_ids]

es_client = Elasticsearch(
    hosts=[os.environ.get('ES_HOST')],
    port=os.environ.get('ES_PORT'),
    http_auth=(os.environ.get('ES_USER'), os.environ.get('ES_PWD')),
    use_ssl=True,
    ca_certs=certifi.where(),
    max_retries=0,
    timeout=30
)


def es_detection_wrapper(
        audiofilename,
        audiofile_starttime,
        audiofile_endtime,
        device_name,
        device_lat,
        device_lon,
        project_name,
        species,
        proba,
        detection_starttime,
        detection_endtime
):
    return {
        "audiofile" : {
            "name" : f"{audiofilename}",
            "startTime" : f"{audiofile_starttime.isoformat()}",
            "endTime" : f"{audiofile_endtime.isoformat()}"
        },
        "device" : {
            "name" : f"{device_name}",
            "status" : "true",
            "location" : {
                "lon" : device_lon,
                "lat" : device_lat
            }
        },
        "project" : f"{project_name}",
        "species" : f"{species}",
        "proba": proba,
        "startTime" : f"{detection_starttime.isoformat()}",
        "endTime" : f"{detection_endtime.isoformat()}",
        "isAuto" : True
    }

def create_div(text, margin_left=0):
    return f'<div style="margin-left:{margin_left}px;">{text}</div>'


def sec2minsec(seconds):
    # convert seconds in minutes, seconds
    return seconds // 60, seconds % 60


def send_to_s3(filepath, key):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_KEY'))
    try:
        response = s3.upload_file(filepath, os.environ.get('AWS_BUCKET'), key)
    except ClientError as e:
        return False, str(e)
    return True, ''


@celery.task(bind=True, name='tasks.process')
def process(self, filename, timestamp_str, chunk_duration, device_name, device_lat, device_lon, project):

    filepath = os.path.join('/uploads', filename)
    timestamp = dateutil.parser.isoparse(timestamp_str)

    msg_list = [f'Splitting files in {chunk_duration}s chunks...']
    self.update_state(state='PROGRESS', meta={'msg': msg_list})

    ######################################################
    # Split file and convert to mono, 16 bits, 22.05 kHz #
    ######################################################

    cp = subprocess.run(
        [
            'ffmpeg',
            '-i', filepath,
            '-ac', '1',
            '-ar', '22050',
            '-sample_fmt', 's16',
            '-f', 'segment',
            '-segment_time', str(chunk_duration),
            f'{os.path.splitext(filepath)[0]}_%03d.wav'
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if cp.returncode != 0:
        os.remove(filepath)
        raise Exception(f'FFMPEG decoding error {filepath.split("/")[-1]}')
    os.remove(filepath)

    ##################
    # Analyse chunks #
    ##################

    # get audio segments
    segments = sorted([s for s in os.listdir('/uploads') if s.startswith(filename[:-4])])

    if not segments:
        raise Exception(f'Chunks not found...')

    tosendtoes = []
    tosendtos3 = []
    for segment in segments:

        # analyse
        msg_list.append(create_div(f'Loading segment {timestamp.time()} -> {(timestamp + timedelta(seconds=chunk_duration)).time()} (device local time)...'))
        self.update_state(state='PROGRESS', meta={'msg': msg_list})
        detections = analyse_file(self, os.path.join('/uploads', segment), msg_list)

        # create unique filename
        unique_filename = str(uuid.uuid4()) + ".wav"
        
        # store data for es
        if detections:
            tosendtoes += [
                es_detection_wrapper(
                    unique_filename,
                    timestamp,
                    timestamp + timedelta(seconds=chunk_duration),
                    device_name,
                    device_lat,
                    device_lon,
                    project,
                    species,
                    proba,
                    timestamp + timedelta(milliseconds=int(times[0]*1000)),
                    timestamp + timedelta(milliseconds=int(times[1]*1000))
                ) for times, species, proba in detections]
    
            pred_str = ''
            for times, species, proba in detections:
                timestamp_start = (timestamp + timedelta(milliseconds=int(times[0]*1000))).time().strftime('%H:%M:%S.%f')[:-3]
                timestamp_end = (timestamp + timedelta(milliseconds=int(times[1]*1000))).time().strftime('%H:%M:%S.%f')[:-3]
                t_str = f'{timestamp_start} -> {timestamp_end}'
                pred_str+=f'{t_str:<30}{species} ({proba:.2f})<br>'

            msg_list.append(create_div(f'{pred_str}', 40))
            self.update_state(state='PROGRESS', meta={'msg': msg_list})

        # store data for s3
        tosendtos3.append((os.path.join('/uploads', segment), unique_filename))
            
        timestamp += timedelta(seconds=chunk_duration)


    #########################
    # Send detections to ES #
    #########################

    msg_list.append(create_div(f'Sending detections to Elastic Search...'))
    res = bulk(es_client, tosendtoes, index=os.environ.get('ES_INDEX'))
    msg_list.append(create_div(f'{res}', 20))
    self.update_state(state='PROGRESS', meta={'msg': msg_list})

    ##########################
    # Send audio files to S3 #
    ##########################

    for filename, key in tosendtos3:
        sent, msg = send_to_s3(os.path.join('/uploads', filename), key)
        if sent:
            os.remove(os.path.join('/uploads', filename))
            msg_list.append(create_div(f'File chunk {filename} sent to AWS S3 successfully.'))
        else:
            msg_list.append(create_div(f'Failed to send file chunk {filename} to AWS S3: {msg}.'))
        self.update_state(state='PROGRESS', meta={'msg': msg_list})
    
    msg_list.append(create_div('Task completed !'))
    return {'msg': msg_list}


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def analyse_file(task, filepath, msg_list):

    audio, sr = load_audio(filepath)

    if audio.shape[1] / sr < AUDIO_SEGMENT_MIN_SIZE:
        msg_list.append(create_div(f'Audio segment is too small (< {AUDIO_SEGMENT_MIN_SIZE}s)', 20))
        task.update_state(state='PROGRESS', meta={'msg': msg_list})
        return

    ##############################################
    # Activity detection and feature computation #
    ##############################################
    
    msg_list.append(create_div('Detecting activity and computing features...', 20))
    task.update_state(state='PROGRESS', meta={'msg': msg_list})

    # compute segment boundaries
    fb_mask = activity_detector.process(audio[0])
    
    if not np.any(fb_mask):
        msg_list.append(create_div('No acoustic activity detected.', 40))
        task.update_state(state='PROGRESS', meta={'msg': msg_list})
        return

    fb_mask_sr = activity_detector.frame_rate
    regions = consecutive(np.where(fb_mask==True)[0])
    boundaries = np.hstack([[r[0],r[-1]] for r in regions]) / activity_detector.frame_rate

    # compute features
    features, mask, times = feature_extractor.process(
        audio,
        sr,
        fb_mask,
        fb_mask_sr,
        min_activity_dur)

    # keep segments with audio activity
    times = [times[i] for i in range(len(times)) if mask[i]]
#    msg_list.append(f'times: {times}')
    features = np.array([features[i] for i in range(features.shape[0]) if mask[i]])

    if features.size == 0:
        msg_list.append(create_div('No acoustic activity detected.', 40))
        task.update_state(state='PROGRESS', meta={'msg': msg_list})
        return

    ##################
    # Bird detection #
    ##################

    msg_list.append(create_div('Detecting bird activity...', 20))
    task.update_state(state='PROGRESS', meta={'msg': msg_list})

    # match bird detection model dimension ordering
    features = np.transpose(features, [0,3,2,1])

    # get bird detection predictions
    headers = {"content-type": "application/json"}
    data = json.dumps({"instances": features.tolist()})
    json_response = requests.post('http://nsb_tf_server:8501/v1/models/bird_detector:predict', data=data, headers=headers)
    predictions = np.array(json.loads(json_response.text)['predictions'])
#    msg_list.append(f'predictions: {predictions}')

    # keep segments with bird activity
    ind = np.where(predictions>BIRD_DETECTION_THRESHOLD)[0]
    times = [times[i] for i in ind]
    features = features[ind]
 #   msg_list.append(f'times: {times}')
 #   task.update_state(state='PROGRESS', meta={'msg': msg_list})

    if features.size == 0:
        msg_list.append(create_div('No bird detected.', 40))
        task.update_state(state='PROGRESS', meta={'msg': msg_list})
        return

    #######################
    # Bird identification #
    #######################

    msg_list.append(create_div('Identifying birds...', 20))
    task.update_state(state='PROGRESS', meta={'msg': msg_list})
    
    # reorder to match bird identification model dimension ordering
    # TODO: build models with same input shape :)
    features = np.transpose(features, [0,2,1,3])

    # normalize
    features = (features - features.mean(axis=(1,2,3), keepdims=True)) / features.std(axis=(1,2,3), keepdims=True)

    # 1 channel -> 3 channels
    features = np.tile(features, [1,1,1,3])
    
    # get bird identification predictions
    headers = {"content-type": "application/json"}
    data = json.dumps({"instances": features.tolist()})
    json_response = requests.post('http://nsb_tf_server:8501/v1/models/bird_classifier:predict', data=data, headers=headers)
    predictions = np.array(json.loads(json_response.text)['predictions'])

    return [(times[ti], target_labels[pi], predictions[ti, pi]) for ti, pi in zip(*np.where(predictions>BIRD_CLASSIFICATION_THRESHOLD))]
