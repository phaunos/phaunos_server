import numpy as np
import librosa

from ..exceptions import NSBAADError


"""Extract mel spectrogram, with linear, log or PCEN scaling.

Mel spectrogram and PCEN parameters values are taken from 

Lostanlen, Vincent, et al.
"Robust sound event detection in bioacoustic sensor networks."
arXiv preprint arXiv:1905.08352 (2019):

and code in
https://github.com/BirdVox/birdvoxdetect/blob/26614955b4019e850599e4a737b0a3ee6341d281/birdvoxdetect/core.py#L817

Audio signal is scaled to [−2**31; 2**31[ with a sample rate of 22.050 Hz
pcen_settings = {
    "fmin": 2000.0,
    "fmax": 11025.0,
    "hop_length": 32,
    "n_fft": 1024,
    "n_mels": 128,
    "pcen_delta": 10.0,
    "pcen_time_constant": 0.06,
    "pcen_norm_exponent": 0.8,
    "pcen_power": 0.25,
    "sr": 22050.0,
    "stride_length": 34,
    "top_freq_id": 128,
    "win_length": 256,
    "window": "flattop"}
"""


LOG_OFFSET = 1e-8


class MelSpectrogramExtractor():

    def __init__(self, config):

        # STFT parameters
        self.sample_rate = config['sample_rate']
        self.win_length = config['win_length']
        self.n_fft = config['n_fft'] 
        self.hop_length = config['hop_length']
        self.window = config['window']

        # Mel spectrogram parameters
        self.n_mels = config['n_mels'] 
        self.fmin = config['fmin'] 
        self.fmax = config['fmax'] 

        self.scaling = config['scaling']
        if self.scaling == 'pcen':
            # PCEN scaling parameters
            self.pcen_eps = config['pcen_eps'] 
            self.pcen_gain = config['pcen_gain'] 
            self.pcen_bias = config['pcen_bias'] 
            self.pcen_power = config['pcen_power'] 
            self.pcen_time_constant = config['pcen_time_constant'] 


    def process(self, audiofile):

        # Open audio file
        audio, sr = librosa.core.load(audiofile)

        # Validate audio
        librosa.util.valid_audio(audio, mono=True)

        if self.scaling == 'pcen':
            # Map to range [-2**31, 2**31[
            audio = (audio * (2**31)).astype('float32')

        # compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            audio,
            sr=self.sample_rate,
            win_length=self.win_length,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            n_mels=self.n_mels,
            htk=True,
            fmin=self.fmin,
            fmax=self.fmax
        )


        if self.scaling == 'pcen':
            # apply PCEN scaling
            return librosa.core.pcen(
                mel_spec,
                sr=sr,
                hop_length=self.hop_length,
                gain=self.pcen_gain,
                bias=self.pcen_bias,
                power=self.pcen_power,
                time_constant=self.pcen_time_constant,
                eps=self.pcen_eps,
                b=None,
                max_size=1,
                ref=None,
                axis=-1,
                max_axis=None
            )
        elif self.scaling == 'log':
            return np.log(mel_spec + LOG_OFFSET)
        elif self.scaling == 'linear':
            return mel_spec
        else:
            raise NSBAADException(f'Scaling {self.scaling} not implemented')

