import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from pydub import AudioSegment

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file, rate=None, data=None):
    if data is None:
        rate, data = get_wav_info(wav_file)
    nfft = 200 # Length of each window segment
    fs = 8000 # Sampling frequencies
    noverlap = 120 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx

# Load a wav file
def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

# Used to standardize volume of audio clip
def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

# Load raw audio files for speech synthesis
def load_raw_audio():
    dialogs = []
    musics = []
    noises = []
    
    for filename in os.listdir("../data/dev_set/dialog"):
        if filename.endswith("wav"):
            try:
                audio = AudioSegment.from_wav("../data/dev_set/dialog/"+filename)
                dialogs.append(audio)
            except Exception:
                print('Error decoding audio file: ', filename)
                
    for filename in os.listdir("../data/dev_set/music"):
        if filename.endswith("wav"):
            try:
                audio = AudioSegment.from_wav("../data/dev_set/music/"+filename)
                musics.append(audio)
            except Exception:
                print('Error decoding audio file: ', filename)
            
    for filename in os.listdir("../data/dev_set/noise"):
        if filename.endswith("wav"):
            try:
                audio = AudioSegment.from_wav("../data/dev_set/noise/"+filename)
                noises.append(audio)
            except Exception:
                print('Error decoding audio file: ', filename)
            
    return dialogs, noises, musics