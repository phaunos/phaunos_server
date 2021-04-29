import numpy as np
import librosa
from scipy.ndimage.morphology import binary_opening
from scipy.ndimage import median_filter

from nsb_aad.base import FrameBasedActivityDetector
from nsb_aad.exceptions import NSBAADError


"""Audio activity detection based on image processing.

The method is described in

Lasseck, Mario. "Bird song classification in field recordings: winning solution for NIPS4B 2013 competition." Proc. of int. symp. Neural Information Scaled for Bioacoustics, sabiod. org/nips4b, joint to NIPS, Nevada. 2013.
"""


def freq2ind(freq, sr, n_fft):
    freq_bin = sr / n_fft 
    return int((freq + freq_bin / 2) / freq_bin)


class MarioDetector(FrameBasedActivityDetector):

    def __init__(self, config):

        # STFT parameters
        self.sample_rate = config['sample_rate']
        self.win_length = config['win_length'] 
        self.hop_length = config['hop_length'] 
        self.min_freq = config.get('min_freq', 0)
        self.max_freq = config.get('max_freq', self.sample_rate / 2)

        if self.min_freq < 0 or self.min_freq > self.sample_rate / 2:
            raise NSBAADError('min_freq must be in [0, sample_rate/2]')
        if self.max_freq < 0 or self.max_freq > self.sample_rate / 2:
            raise NSBAADError('max_freq must be in [0, sample_rate/2]')
        if self.max_freq <=  self.min_freq:
            raise NSBAADError('max_freq must be greater than min_freq')

        self.min_spec_ind = freq2ind(self.min_freq, self.sample_rate, self.win_length )
        self.max_spec_ind = freq2ind(self.max_freq, self.sample_rate, self.win_length )

        # Detection parameters
        self.clipping_threshold = config['clipping_threshold']
        self.opening_kernel_shape = config['opening_kernel_shape']
        self.median_filter_shape = config['median_filter_shape']

    @property
    def frame_rate(self):
        return self.sample_rate / self.hop_length


    def process(self, audio):

        # compute magnitude spectrogram
        spec = np.abs(librosa.stft(
            audio,
            n_fft=self.win_length,
            hop_length=self.hop_length,
            center=False
        ))[self.min_spec_ind:self.max_spec_ind,:]

        # binarize data
        col_median = np.median(spec, axis=0, keepdims=True)
        row_median = np.median(spec, axis=1, keepdims=True)
        spec[spec < row_median * self.clipping_threshold] = 0
        spec[spec < col_median * self.clipping_threshold] = 0
        spec[spec > 0] = 1

        # opening (erosion + dilation)
        spec = binary_opening(spec, structure=np.ones(self.opening_kernel_shape))

        # merge detection along the frequency axis
        spec = np.any(spec, axis=0)

        # median filtering to remove short isolated detection
        return median_filter(spec, size=self.median_filter_shape)

