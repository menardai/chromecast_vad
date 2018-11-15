import os
import gc
import numpy as np

from random import randint
from pydub import AudioSegment
from vad_model import VadModel

from keras.utils import Sequence


class SpectrogramDataGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        '''
        Here, `x_set` is list of path to the spectrogram .npy file
        and `y_set` are the associated truth vector .npy file
        '''
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        ''' this method should return a complete batch. '''
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return (np.array([np.load(filename) for filename in batch_x]),
                np.array([np.load(filename) for filename in batch_y]))

    def on_epoch_end(self):
        ''' If you want to modify your dataset between epochs you may implement. '''
        pass


class Dataset(object):

    def __init__(self, Tx, Ty, n_freq, dialog_dir, noise_dir, music_dir, verbose=False):
        """
        Arguments:
            Tx - integer, The number of time steps input to the model from the spectrogram
            Ty: integer, The number of time steps in the output of our model
            n_freq - integer, Number of frequencies input to the model at each time step of the spectrogram
            dialog_dir - string, path to dialog wav files
            noise_dir - string, path to dialog wav files
            music_dir - string, path to dialog wav files
            verbose - boolean, debug verbose flag
        """
        self.Tx = Tx
        self.Ty = Ty
        self.n_freq = n_freq
        self.verbose = verbose

        self.dialogs, self.noises, self.musics  = self._load_raw_audio(dialog_dir, noise_dir, music_dir)

    def create_dev_dataset(self, n_x, output_x_filename=None, output_y_filename=None):
        X = np.zeros((n_x, self.Tx, self.n_freq))
        Y = np.zeros((n_x, self.Ty, 1))

        print('number of training samples to generate =', n_x)

        for i in range(n_x-1):
            if i % 100 == 0:
                print('sample {0}/{1}'.format(i, n_x))
                gc.collect()

            music_index = randint(0, len(self.musics)-1)
            noise_index = randint(0, len(self.noises)-1)

            x, y = self.create_training_example(self.musics[music_index], self.dialogs, self.noises[noise_index], verbose=False)

            X[i] = x.transpose()
            Y[i] = y.transpose()

        if output_x_filename is not None:
            if self.verbose: print('saving dev set X ...')
            np.save(output_x_filename, X)

        if output_y_filename is not None:
            if self.verbose: print('saving dev set Y ...')
            np.save(output_y_filename, Y)

        return X, Y

    def create_dev_dataset_files(self, n_x, output_folder, start_index=0):
        print('number of training samples to generate =', n_x)
        voice_sample_count = 0

        for i in range(start_index, start_index + n_x):
            if i % 100 == 0:
                print('sample {0}/{1}'.format(i, n_x))
                gc.collect()

            music_index = randint(0, len(self.musics)-1)
            noise_index = randint(0, len(self.noises)-1)

            x_wav_filename = '{}/x_{}.wav'.format(output_folder, i)
            x_sample_filename = '{}/x_spectrogram_{}.npy'.format(output_folder, i)
            y_sample_filename = '{}/y_{}.npy'.format(output_folder, i)

            x, y = self.create_training_example(self.musics[music_index], self.dialogs, self.noises[noise_index],
                                                output_wav_filename=x_wav_filename, verbose=False)
            X = x.transpose()
            Y = y.transpose()
            np.save(x_sample_filename, X)
            np.save(y_sample_filename, Y)

            if y.argmax() > 0: voice_sample_count += 1
            del X, Y, x, y

        print('voice_sample_count = ', voice_sample_count)

    def load_dataset(x_filename, y_filename):
        X = np.load(x_filename)
        Y = np.load(y_filename)
        return X, Y

    def create_training_example(self, music, dialogs, noise, output_wav_filename='train.wav', verbose=False):
        """
        Creates a training example with a given music, noise, and dialog.

        Arguments:
        music -- a 10 second music audio recording
        dialogs -- a list of audio segments of a conversation between two persons
        noise -- a 10 second noise audio recording

        Returns:
        x -- the spectrogram of the training example
        y -- the label at each time step of the spectrogram
        """

        dB_reduction = np.random.randint(5, 20)
        noise = noise - dB_reduction

        if np.random.randint(0, 10) == 0:
            # 10% of the time, only noise (no music)
            mixed_audio = noise
            if verbose: print("noise {0} dB".format(dB_reduction))
        else:
            # Make music quieter or louder
            dB_reduction = np.random.randint(-5, 20)
            mixed_audio = music + dB_reduction
            if verbose: print("music {0} dB".format(dB_reduction))

            # insert the noise audio over mixed_audio and optional dialog
            mixed_audio = mixed_audio.overlay(noise, position=0)

        # Initialize y (label vector) of zeros
        y = np.zeros((1, self.Ty))

        # add a voice 9/10 times
        if np.random.randint(0, 10) != 0:
            # Select random "dialog" audio clips from the entire list of "dialogs" recordings
            number_of_dialogs = 1  # np.random.randint(0, 2)
            random_indices = np.random.randint(len(dialogs), size=number_of_dialogs)
            random_dialogs = [dialogs[i] for i in random_indices]

            # Loop over randomly selected "conversation" clips and insert in mixed_audio
            for random_dialog in random_dialogs:
                # Make dialog quieter or louder
                dB_reduction = np.random.randint(0, 10)
                random_dialog = random_dialog + dB_reduction

                # Insert the audio clip on the mixed_audio
                if verbose: print("dialog insertion... {0} dB".format(dB_reduction))
                mixed_audio, segment_time = self._insert_audio_clip(mixed_audio, random_dialog)
                # Retrieve segment_start and segment_end from segment_time
                segment_start, segment_end = segment_time
                # Insert labels in "y"
                y = self._insert_ones(y, segment_start, segment_end)
                if verbose: print("dialog inserted [{0}, {1}]".format(segment_start, segment_end))

        # Standardize the volume of the audio clip
        mixed_audio = self._match_target_amplitude(mixed_audio, -20.0)

        # Export new training example
        file_handle = mixed_audio.export(output_wav_filename, format="wav")
        file_handle.close()
        del mixed_audio

        # Get and plot spectrogram of the new recording (mixed_audio with superposition of music, noise and dialog)
        x = VadModel.graph_spectrogram(output_wav_filename)
        if verbose: print("-----------------------")

        return x, y

    def _load_raw_audio(self, dialog_dir, noise_dir, music_dir):
        ''' Load raw audio files. '''
        i = 0
        loading_list = (
            {
                'dir': dialog_dir,
                'list': []
            },
            {
                'dir': noise_dir,
                'list': []
            },
            {
                'dir': music_dir,
                'list': []
            }
        )

        for loading in loading_list:
            for filename in os.listdir(loading['dir']):
                if filename.endswith("wav"):
                    try:
                        audio = AudioSegment.from_wav('{}/{}'.format(loading['dir'], filename))
                        loading['list'].append(audio)

                        if self.verbose and i % 500 == 0: print('raw audio count = {0}'.format(i))
                        i += 1
                    except Exception:
                        print('Error decoding audio file: ', filename)

        return loading_list[0]['list'], loading_list[1]['list'], loading_list[2]['list']

    def _insert_audio_clip(self, background, audio_clip, previous_segments=None):
        """
        Insert a new audio segment over the background noise at a random time step, ensuring that the
        audio segment does not overlap with existing segments.

        Arguments:
        background -- a 10 second background audio recording.
        audio_clip -- the audio clip to be inserted/overlaid.
        previous_segments -- times where audio segments have already been placed; None

        Returns:
        new_background -- the updated background audio
        """
        # Get the duration of the audio clip in ms
        segment_ms = len(audio_clip)

        # Use one of the helper functions to pick a random time segment onto which to insert
        # the new audio clip.
        segment_time = self._get_random_time_segment(segment_ms)

        # Check if the new segment_time overlaps with one of the previous_segments. If so, keep
        # picking new segment_time at random until it doesn't overlap.
        if previous_segments is not None:
            while self._is_overlapping(segment_time, previous_segments):
                segment_time = self._get_random_time_segment(segment_ms)
                # Add the new segment_time to the list of previous_segments
            previous_segments.append(segment_time)

        # Superpose audio segment and background
        new_background = background.overlay(audio_clip, position = segment_time[0])

        return new_background, segment_time

    def _insert_ones(self, y, segment_start_ms, segment_end_ms):
        """
        Update the label vector y. The labels of the segment's output steps should be set to 1.

        Arguments:
        y -- numpy array of shape (1, Ty), the labels of the training example
        segment_start_ms -- the start time of the segment in ms
        segment_end_ms -- the end time of the segment in ms

        Returns:
        y -- updated labels
        """
        # duration of the background (in terms of spectrogram time-steps)
        segment_start_y = int(segment_start_ms * self.Ty / 10000.0)
        segment_end_y = int(segment_end_ms * self.Ty / 10000.0)

        # Add 1 to the correct index in the background label (y)
        for i in range(segment_start_y, segment_end_y + 1):
            if i < self.Ty:
                y[0, i] = 1

        return y

    def _get_random_time_segment(self, segment_ms):
        """
        Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

        Arguments:
        segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

        Returns:
        segment_time -- a tuple of (segment_start, segment_end) in ms
        """
        # Make sure segment doesn't run past the 10sec background
        segment_start = np.random.randint(low=0, high=10000-segment_ms)
        segment_end = segment_start + segment_ms - 1

        return (segment_start, segment_end)

    def _is_overlapping(self, segment_time, previous_segments):
        """
        Checks if the time of a segment overlaps with the times of existing segments.

        Arguments:
        segment_time -- a tuple of (segment_start, segment_end) for the new segment
        previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

        Returns:
        True if the time segment overlaps with any of the existing segments, False otherwise
        """
        segment_start, segment_end = segment_time

        # Step 1: Initialize overlap as a "False" flag.
        overlap = False

        # Step 2: loop over the previous_segments start and end times.
        # Compare start/end times and set the flag to True if there is an overlap
        for previous_start, previous_end in previous_segments:
            if segment_start <= previous_end and segment_end >= previous_start:
                overlap = True

        return overlap

    def _match_target_amplitude(self, sound, target_dBFS):
        ''' Used to standardize volume of audio clip. '''
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)


if __name__ == '__main__':
    dataset = Dataset(Tx=5511, Ty=1375, n_freq=101,
                      dialog_dir='data/dev_set_wav/dialog',
                      noise_dir='data/dev_set_wav/noise',
                      music_dir='data/dev_set_wav/music',
                      verbose=True)

    print("music[0]: " + str(len(dataset.musics[0])))        # Should be 10,000, since it is a 10 sec clip
    print("dialogs[0]: " + str(len(dataset.dialogs[0])))     # Between 3000 and 9000 (random length)
    print("noises[0]: " + str(len(dataset.noises[0])))       # Should be 10,000, since it is a 10 sec clip

    print('music 10s audio count = ', len(dataset.musics))
    print('dialogs audio count = ', len(dataset.dialogs))
    print('noises 10s audio count = ', len(dataset.noises))

    # x, y = dataset.create_training_example(dataset.musics[1], dataset.dialogs, dataset.noises[1])
    # print('x.shape =', x.shape)
    # print('y.shape =', y.shape)

    #dataset.create_dev_dataset(2500, '../data/dev_set_2500_x.npy', '../data/dev_set_2500_y.npy')
    #dataset.create_dev_dataset_files(5000, 'data/dev_set')
    dataset.create_dev_dataset_files(5000, 'data/dev_set', start_index=5000)
