# Nature Sound Box Audio Activity Detection

## Install / activate environment

Clone repository

`$ git clone git@bitbucket.org:securaxisteam/nsb_aad.git`

Create conda environment (to install Anaconda, see https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

`$ conda env create -f nsb_aad/environment.yml`

Activate environment

`$ conda activate nsb_aad`

Add package to PYTHONPATH

`$ export PYTHONPATH=$PYTHONPATH:/path/to/nsb_aad/`

## Run detection script

```
$ python /path/to/nsb_aad/nsb_aad/segment_based_detectors/nsb_detector.py --help
usage: nsb_detector.py [-h] [--config_mario CONFIG_MARIO]
                       [--config_nsb CONFIG_NSB]
                       audiofilename outfilename

positional arguments:
  audiofilename         Input audio file.
  outfilename           Output CSV file.

optional arguments:
  -h, --help            show this help message and exit
  --config_mario CONFIG_MARIO
                        Configuration file for mario detector.
  --config_nsb CONFIG_NSB
                        Configuration file for nsb detector.
```

In the example below, the audio file must be @ 22.05 kHz (otherwise set your own config files)

`$  python /path/to/nsb_aad/nsb_aad/segment_based_detectors/nsb_detector.py somefile.wav out.csv`

## Check algorithm description in notebook

The algorithm is described with an example in https://bitbucket.org/securaxisteam/nsb_aad/src/master/notebooks/mario_detector.ipynb?viewer=nbviewer.
