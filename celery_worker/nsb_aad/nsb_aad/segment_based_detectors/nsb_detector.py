import os
import argparse
import json
import numpy as np
import librosa

from nsb_aad.base import SegmentBasedActivityDetector
from nsb_aad.frame_based_detectors.mario_detector import MarioDetector
from nsb_aad.exceptions import NSBAADError


"""Integration of short-term detections over longer audio segments.

A segment is labeled as "containing activity" if it contains short-term activity.
"""


class NSBDetector(SegmentBasedActivityDetector):

    def __init__(self, config):

        # Short-term frame parameters
        self.sample_rate = config['sample_rate']
        self.frame_hop_length = config['frame_hop_length'] 

        # Segment parameters
        self.segment_duration = config['segment_duration']
        self.segment_hop_duration = config['segment_hop_duration']

        if self.segment_hop_duration <= 0 or self.segment_hop_duration > self.segment_duration:
            raise NSBAADError('segment_hop_duration must be in ]0, segment_duration]')


    def process(self, detections, detection_sr):

        # Compute segment boundaries and check whether each segment contain a detection

        times = []
        labels = []
        start = 0
        while start + self.segment_duration <= len(detections) / detection_sr:
            end = start + self.segment_duration
            times.append((start, end))
            labels.append(np.any(detections[int(start*detection_sr):int(end*detection_sr)]))
            start += self.segment_hop_duration

        return times, labels


def run(audiofilename, outfilename, configfilename_mario, configfilename_nsb):

    # get default config if not specified
    path = os.path.dirname(__file__)
    if not configfilename_mario:
        configfilename_mario = os.path.join(path, '../../configs/mario.json')
    if not configfilename_nsb:
        configfilename_nsb = os.path.join(path, '../../configs/nsb.json')

    # parse config files
    config_mario = json.load(open(configfilename_mario, 'r'))
    config_nsb = json.load(open(configfilename_nsb, 'r'))

    # open audio file
    audio, _ = librosa.load(
        audiofilename,
        sr=config_mario['sample_rate'],
        mono=True)

    # frame-based detection
    fb_detector = MarioDetector(config_mario)
    fb_detections = fb_detector.process(audio)

    # segment-based detection
    sb_detector = NSBDetector(config_nsb)
    times, labels = sb_detector.process(fb_detections, fb_detector.frame_rate)

    # write detections in csv file
    with open(outfilename, 'w') as outfile: 
        for i in range(len(times)):
            outfile.write(f'{times[i][0]:.3f},{times[i][1]:.3f},{int(labels[i])}\n')


if __name__ == '__main__':

    """Label segments defined in config_nsb as
        - not containing audio activity (0)
        - containing audio activity (1)

    The results are written to outfilename as a CSV file with the following format:

    ***outfile.csv***
    <segment_start_time>,<segment_start_time>,<label>
    <segment_start_time>,<segment_start_time>,<label>
    ...
    *****************
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('audiofilename', type=str,
                        help='Input audio file.')
    parser.add_argument('outfilename', type=str,
                        help='Output CSV file.')
    parser.add_argument('--config_mario', type=str,
                        help='Configuration file for mario detector.')
    parser.add_argument('--config_nsb', type=str,
                        help='Configuration file for nsb detector.')
    args = parser.parse_args()

    run(
        args.audiofilename,
        args.outfilename,
        args.config_mario,
        args.config_nsb,
    )

