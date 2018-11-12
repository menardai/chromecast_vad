import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.io import wavfile
#from pydub import AudioSegment

from keras.models import model_from_json
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape


class VadModel(object):

    def __init__(self, input_shape=(5511, 101), architecture_filename=None, weights_filename=None):
        if architecture_filename:
            # Model reconstruction from JSON file
            with open(architecture_filename, 'r') as f:
                self.model = model_from_json(f.read())
        else:
            self.model = VadModel.get_model(input_shape)

        if weights_filename:
            self.model.load_weights(weights_filename)

        self.version = 'v1.0.0'
        self.model.summary()

    def get_model(input_shape):
        """
        Function creating the model's graph in Keras.

        Argument:
        input_shape -- shape of the model's input data (using Keras conventions)

        Returns:
        model -- Keras model instance
        """
        X_input = Input(shape=input_shape)

        # CONV layer
        X = Conv1D(196, 15, strides=4)(X_input)  # CONV1D
        X = BatchNormalization()(X)              # Batch normalization
        X = Activation('relu')(X)                # ReLu activation
        #X = Dropout(0.8)(X)

        # First GRU Layer
        X = GRU(128, return_sequences=True)(X)   # GRU (use 128 units and return the sequences)
        X = Dropout(0.5)(X)                      # dropout
        X = BatchNormalization()(X)              # Batch normalization

        # Second GRU Layer
        X = GRU(128, return_sequences=True)(X)   # GRU (use 128 units and return the sequences)
        X = Dropout(0.5)(X)                      # dropout
        X = BatchNormalization()(X)              # Batch normalization
        X = Dropout(0.5)(X)                      # dropout

        # Time-distributed dense layer
        X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

        model = Model(inputs = X_input, outputs = X)

        return model

    def predict(self, wav_filename=None, rate=None, data=None, show_graphs=False):
        if show_graphs:
            plt.subplot(2, 1, 1)  # spectrogram on top

        x = VadModel.graph_spectrogram(wav_filename, rate=rate, data=data)

        # the spectogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
        x = x.swapaxes(0,1)
        x = np.expand_dims(x, axis=0)

        predictions = self.model.predict(x)

        if show_graphs:
            plt.subplot(2, 1, 2)  # probability plot at the bottom
            plt.plot(predictions[0,:,0])
            plt.ylabel('probability')
            plt.show()

        return predictions

    def graph_spectrogram(wav_filename, rate=None, data=None):
        ''' Calculate and plot spectrogram for a wav audio file. '''
        if data is None:
            rate, data = VadModel.get_wav_info(wav_filename)
        nfft = 200 # Length of each window segment
        fs = 8000 # Sampling frequencies
        noverlap = 120 # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        return pxx

    def get_wav_info(wav_filename):
        ''' Load a wav file. '''
        rate, data = wavfile.read(wav_filename)
        return rate, data


if __name__ == '__main__':
    print('loading model...')
    vad_model = VadModel(architecture_filename='models/model_architecture.json', weights_filename='models/vad_09_11_2018_weights.h5')

    # Save the model architecture
    #with open('models/model_architecture.json', 'w') as f:
    #    f.write(vad_model.model.to_json())

    print('loading wav file...')
    wav_filename = 'data/test_set_wav/test_with_dialog_00105.wav'
    rate, data = VadModel.get_wav_info(wav_filename)

    print('waiting 2 seconds...')
    time.sleep(2)

    print('starting prediction for the loaded audio file...')
    start = time.time()

    predictions = vad_model.predict(wav_filename=None, rate=rate, data=data, show_graphs=False)
    #predictions = vad_model.predict(wav_filename=wav_filename)

    end = time.time()

    print('prediction time = {0:01f}s'.format(end - start))
    print('prediction length:', len(predictions[0]))
    print('prediction:', str(predictions[0]))

