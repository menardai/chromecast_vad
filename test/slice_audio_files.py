import unittest
import glob
import shutil

from pydub import AudioSegment
from preprocessing.audioslice import AudioSlice


class SpliceAudioFiles(unittest.TestCase):

    def test_init_params(self):
        # fixed length
        a = AudioSlice('sliced_music', 'background_music_', fixed_length_s=10)
        self.assertEqual(a.fixed_sequence_length_ms, 10 * 1000)

        # random length in a range
        b = AudioSlice('sliced_music', 'background_music_', min_length_s=5, max_length_s=8)
        self.assertEqual(b.min_length_ms, 5 * 1000)
        self.assertEqual(b.max_length_ms, 8 * 1000)

        # test min without max
        with self.assertRaises(RuntimeError):
            AudioSlice('sliced_music', 'background_music_', min_length_s=5)

        # test max without min
        with self.assertRaises(RuntimeError):
            AudioSlice('sliced_music', 'background_music_', max_length_s=5)

        # test min larger than max is raising an error
        with self.assertRaises(RuntimeError):
            AudioSlice('sliced_music', 'background_music_', min_length_s=10, max_length_s=8)

    def test_chunk_fixed_length(self):
        """
        The 168.96s mp3 file is sliced into 16 chunks of 10s each. The last 8s are drop.
        """
        song = AudioSlice('sliced_music', 'background_music_', fixed_length_s=10)
        next_index = song.save_chunks('music/Isabelle.mp3', 1, verbose=True)
        self.assertEqual(next_index, 17)

    def test_chunk_start_time(self):
        """
        The 168.96s mp3 file is reduce to 48.96s with a starting time at 120s. The result is 4 chunks of 10s each.
        """
        song = AudioSlice('sliced_music', 'background_music_', fixed_length_s=10, start_time_s=120, output_format='wav')
        next_index = song.save_chunks('music/Isabelle.mp3', 100, verbose=True)
        self.assertEqual(next_index, 104)

    def test_chunk_random_length_basic(self):
        song = AudioSlice('sliced_conversation', 'conversation_', min_length_s=4, max_length_s=9)
        next_index = song.save_chunks('conversation/talk_01_60s.mp3', 1, verbose=True)
        self.assertNotEqual(next_index, 1)

    def test_chunk_random_length_size(self):
        # delete destination folder cause we will read its contents as a test
        shutil.rmtree('sliced_conversation')

        song = AudioSlice('sliced_conversation', 'dialog_', min_length_s=4, max_length_s=9, start_time_s=30)

        song.save_chunks('conversation/talk_01_60s.mp3', 1, verbose=True)
        #song.save_chunks('../raw_data/dialog/autremid-20160819-1254 Anna Casabonne et Charles-Antoine Crête.mp3', 1, verbose=True)

        dialog_filenames = glob.glob('sliced_conversation/dialog_*')
        for dialog_filename in dialog_filenames:
            audio = AudioSegment.from_wav(dialog_filename)

            self.assertGreaterEqual(audio.duration_seconds, 4)
            self.assertLessEqual(audio.duration_seconds, 9)


if __name__ == '__main__':
    unittest.main()