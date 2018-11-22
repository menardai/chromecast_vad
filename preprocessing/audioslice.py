import os

from pydub import AudioSegment
from random import randint


class AudioSlice(object):
    """
    Split a list of audio files in fixed length sequence.
    Save each sequence as a separate mp3 file.
    """

    def __init__(self,
                 output_dir, output_filename_prefix,
                 start_time_s=None,
                 fixed_length_s=None,
                 min_length_s=None, max_length_s=None,
                 min_db=None,
                 output_format='wav'):
        """
        Arguments:
            output_dir -- string, output folder name
            output_filename_prefix -- string, chunk prefix filename
            start_time_s -- integer, start reading audio at 'start_time_s' seconds
            fixed_length_s -- integer, a fixed sequence length in seconds
            min_length_s -- integer, random segment length minimum value in seconds
            max_length_s -- integer, random segment length maximum value in seconds
        """
        self.start_time_ms = start_time_s * 1000 if start_time_s else 0
        self.output_format = output_format

        self.output_dir = output_dir
        self.output_filename_prefix = output_filename_prefix
        self.min_db = min_db

        # create output folder if not exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if fixed_length_s:
            self.fixed_sequence_length_ms = 1000 * fixed_length_s
        else:
            self.fixed_sequence_length_ms = None

            if min_length_s and max_length_s:
                if min_length_s > 0 and max_length_s >= min_length_s:
                    self.min_length_ms = 1000 * min_length_s
                    self.max_length_ms = 1000 * max_length_s
                else:
                    raise RuntimeError('Bad constructor arguments. '
                                       'Max must be greater than min value.')
            else:
                raise RuntimeError('Bad constructor arguments. '
                                   'Must have a fixed length or a min and max.')

    def save_chunks(self, audio_source_filename, next_chunk_index, verbose=False):
        """
        Arguments:
            audio_source_filename -- string, path to the mp3 file to slice
            next_chunk_index -- integer, the chunk index to be append to output_filename_prefix

        Returns:
            integer, the following chunk index.
        """

        # load source audio file
        if verbose: print(audio_source_filename)
        if audio_source_filename.endswith('.mp3'):
            audio = AudioSegment.from_mp3(audio_source_filename)
        else:
            audio = AudioSegment.from_wav(audio_source_filename)

        if verbose: print('{} seconds'.format(audio.duration_seconds))

        # shorthen the audio clip if a specific starting time has been defined
        if self.start_time_ms:
            audio = audio[self.start_time_ms:]
            if verbose: print('duration after update of starting time = {} seconds'.format(audio.duration_seconds))

        # set frame rate and set mono channel for wav output
        if self.output_format == 'wav':
            audio = audio.set_frame_rate(44100)
            audio = audio.set_channels(1)

        if self.fixed_sequence_length_ms is not None:
            return self._save_chunks_fixed_length(audio, next_chunk_index, verbose)
        else:
            return self._save_chunks_random_length(audio, next_chunk_index, verbose)

    def _save_chunks_fixed_length(self, audio, next_chunk_index, verbose=False):
        if verbose: print('slicing in {}s chunks'.format(self.fixed_sequence_length_ms / 1000))

        # split sound in XX seconds slices and save on disk
        for i, chunk in enumerate(audio[::self.fixed_sequence_length_ms]):
            if chunk.duration_seconds == (self.fixed_sequence_length_ms / 1000):
                if self.min_db is None or chunk.max_dBFS >= self.min_db:
                    chunk_name = "{0}/{1}{2:05d}.{3}".format(self.output_dir, self.output_filename_prefix, next_chunk_index, self.output_format)
                    if verbose: print("exporting", chunk_name)

                    with open(chunk_name, "wb") as f:
                        chunk.export(f, format=self.output_format)

                    next_chunk_index += 1

        return next_chunk_index

    def _save_chunks_random_length(self, audio, next_chunk_index, verbose=False):
        if verbose: print('slicing in random length chunks [{0}, {1}]'.format(self.min_length_ms / 1000, self.max_length_ms / 1000))

        remaining_time_ms = audio.duration_seconds * 1000

        start_ms = 0
        length_ms = randint(self.min_length_ms, self.max_length_ms)

        # special case for audio clip shorter than selected random length.
        # ex: audio of 6s, random length of 8s --> will reset the random length to 6s
        if remaining_time_ms < length_ms and remaining_time_ms > self.min_length_ms:
            length_ms = remaining_time_ms - 100

        while remaining_time_ms - length_ms > 0:
            chunk_name = "{0}/{1}{2:05d}.{3}".format(self.output_dir, self.output_filename_prefix, next_chunk_index, self.output_format)
            if verbose: print("exporting {0}ms - {1}".format(length_ms, chunk_name))

            chunk = audio[start_ms: start_ms+length_ms]
            if verbose: print('chunk {}s'.format(chunk.duration_seconds))

            with open(chunk_name, "wb") as f:
                chunk.export(f, format=self.output_format)

            next_chunk_index += 1
            start_ms += length_ms
            remaining_time_ms -= length_ms
            length_ms = randint(self.min_length_ms, self.max_length_ms)

        return next_chunk_index







