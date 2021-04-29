import numpy as np
import librosa

from ..exceptions import NSBAADError


"""Extract STFT power spectrogram"""


LOG_OFFSET = 1e-8


class STFTSpectrogramExtractor():

    def __init__(self, config):

        # STFT parameters
        self.sample_rate = config['sample_rate']
        self.win_length = config['win_length']
        self.n_fft = config.get('n_fft', self.win_length)
        self.hop_length = config['hop_length']
        self.window = config.get('window', 'hann')

        self.scaling = config['scaling']


    def process(self, audiofile):

        # Open audio file
        audio, sr = librosa.core.load(audiofile)

        # Validate audio
        librosa.util.valid_audio(audio, mono=True)

        # compute stft-spectrogram
        stft_spec = np.abs(librosa.core.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window
        )) ** 2


        if self.scaling == 'linear':
            return stft_spec
        elif self.scaling == 'log':
            return np.log(stft_spec + LOG_OFFSET)
        else:
            raise NSBAADException(f'Scaling {self.scaling} not implemented')

