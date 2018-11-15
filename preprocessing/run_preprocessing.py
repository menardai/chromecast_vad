import glob

from preprocessing.audioslice import AudioSlice


# MUSIC
#
# convert all mp3 files from raw_music folder into 10s wav file (saved in data folder):
#   raw_data/music/*.mp3 --> data/music/music_xxxxx.wav
#
music = AudioSlice(output_dir='../data/dev_set_wav/music', output_filename_prefix='music_', fixed_length_s=10)

raw_mp3_files = glob.glob('../raw_data/music_mp3/*.mp3')
next_index = 0

for mp3_file in raw_mp3_files:
    try:
        next_index = music.save_chunks(audio_source_filename=mp3_file, next_chunk_index=next_index, verbose=True)
    except Exception:
        print("Can't decode music mp3 file:", mp3_file)
#
#
# # NOISE
# noise = AudioSlice(output_dir='../data/dev_set_wav/noise', output_filename_prefix='noise_', fixed_length_s=10)
#
# raw_noise_wav_files = glob.glob('../raw_data/noise/*.wav')
# next_index = 0
#
# for noise_file in raw_noise_wav_files:
#     try:
#         next_index = noise.save_chunks(audio_source_filename=noise_file, next_chunk_index=next_index, verbose=True)
#     except Exception:
#         print("Can't decode noise audio file:", noise_file)
#
#
# CONVERSATIONS
#dialog = AudioSlice(output_dir='../data/dev_set_wav/dialog', output_filename_prefix='dialog_',
#                    min_length_s=4, max_length_s=9, start_time_s=120)
# dialog = AudioSlice(output_dir='../data/dev_set_wav/dialog', output_filename_prefix='dialog_',
#                     min_length_s=3, max_length_s=9)
#
# raw_dialog_files = glob.glob('../raw_data/dialog/*.mp3')
# next_index = 0
#
# for dialog_file in raw_dialog_files:
#     try:
#         next_index = dialog.save_chunks(audio_source_filename=dialog_file, next_chunk_index=next_index, verbose=True)
#     except Exception:
#         print("Can't decode dialog mp3 file:", dialog_file)


# TEST SET -- CONVERSATIONS
# test = AudioSlice(output_dir='../data/test_set', output_filename_prefix='test_with_dialog_', fixed_length_s=10, start_time_s=13)
# next_index = test.save_chunks(audio_source_filename='../raw_data/test_set/with_dialog_01.wav', next_chunk_index=100, verbose=True)
#
# test = AudioSlice(output_dir='../data/test_set', output_filename_prefix='test_without_dialog_', fixed_length_s=10, start_time_s=13)
# next_index = test.save_chunks(audio_source_filename='../raw_data/test_set/without_dialog_01.wav', next_chunk_index=400, verbose=True)
