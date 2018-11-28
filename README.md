# chromecast_vad
Keras RNN implementation of a voice activity detector to control Chromecast device volume.

## Dependencies
- [Pychromecast](https://github.com/balloob/pychromecast)
- [Pydub](http://pydub.com/)

## Preprocessing
**python run_preprocessing.py**:  
Convert various length mp3/wav files into 2 seconds wav files (audio sampled at 44100 Hz, mono channel).

## Training Set
**python dataset.py**  
Convert preprocessed audio files in samples of 2 seconds wav files, X and Y numpy arrays.  
- X is the numpy array of a spectrogram with 101 frequencies.  
- Y is a 0/1 numpy array (speech or not)

## Model Training 
**python run_experiments.py**

## Running
**Build the docker image of the Flask *app.py* web service.**  
This web service take a 2s audio file as input, use the RNN model to predict there is speech in the audio files and 
returns a true/false prediction.

**docker_build_image.sh**  
Create a Docker image with app.py over ufoym/deepo:keras-py36-cpu (a python 3.6, Keras on CPU image).

**docker_run_webapp.sh**  
To start the web server we just built.

**python chromecast_live_volume.py**  
To start listening and controlling the volume of the chromecast device of your choice.

## Dataset
A synthesized dataset created from merging background noise, music and speech.
- [Common Voice](https://voice.mozilla.org/en/datasets) by Mozilla  
an open and publicly available dataset of voices that everyone can use to train speech-enabled applications.
- [QUT-NOISE-SRE](https://research.qut.edu.au/saivt/databases/qut-noise-databases-and-protocols/) Databases  
D. Dean, A. Kanagasundaram, H. Ghaemmaghami, M. Hafizur, S. Sridharan (2015) “The QUT-NOISE-SRE protocol for the evaluation of noisy speaker recognition”. In Proceedings of Interspeech 2015, September, Dresden, Germany.
- Music played by a home speaker recorded using Audacity

## Live Demo


## Live Demo Output
```
$ python chromecast_live_volume.py

Looking for chromecast devices...

Connected to: Cuisine
initial volume = 0.45
no music playing
no music playing
no music playing
no music playing
> recording
  speech probability = 0.01
  steps_without_speech = 1
> recording
  speech probability = 0.01
  steps_without_speech = 2
> recording
  speech probability = 0.00
  steps_without_speech = 3
> recording
  speech probability = 0.05
  steps_without_speech = 4
> recording
  speech probability = 1.00
  *** SPEECH DETECTED ***
  set volume to 0.30
> recording
  speech probability = 1.00
  *** SPEECH DETECTED ***
> recording
  speech probability = 1.00
  *** SPEECH DETECTED ***
> recording
  speech probability = 1.00
  *** SPEECH DETECTED ***
> recording
  speech probability = 1.00
  *** SPEECH DETECTED ***
> recording
  speech probability = 0.09
  steps_without_speech = 1
> recording
  speech probability = 0.07
  steps_without_speech = 2
> recording
  speech probability = 0.03
  steps_without_speech = 3
  set volume to 0.45
> recording
  speech probability = 0.01
  steps_without_speech = 4
> recording
  speech probability = 0.01
  steps_without_speech = 5
```

