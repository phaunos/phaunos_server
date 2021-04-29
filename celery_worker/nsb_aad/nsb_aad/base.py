from abc import ABC, abstractmethod


# Used as a template...
class FrameBasedActivityDetector(ABC):

    @abstractmethod
    def __init__(self, config=None):
        """Possibly get some config"""

    @abstractmethod
    def process(self, audiofile):
        """Take an audio file and return
        short-term detections"""


class SegmentBasedActivityDetector(ABC):

    @abstractmethod
    def __init__(self, config=None):
        """Possibly get some config"""

    @abstractmethod
    def process(self, data):
        """Take short-term detections (array of bool) and return
        segment boundaries and binary label (0: no activity,
        1: activity)"""
