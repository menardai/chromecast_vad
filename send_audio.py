import requests
import json

import numpy as np


url = 'http://127.0.0.1:5000/predict'

files = {'audio': open('notebooks/audio_examples/live_speech_2s_00001.wav', 'rb')}
#files = {'audio': open('notebooks/audio_examples/live_speech_2s_00002.wav', 'rb')}
#files = {'audio': open('notebooks/audio_examples/test_music_only.wav', 'rb')}
#files = {'audio': open('data/live_recording.wav', 'rb')}

r = requests.post(url, files=files)
#print(r.text)

result_json = json.loads(r.text)
if result_json['success']:
    # create a numpy array from json
    p = np.array(result_json['predictions'])

    print('probability someone is talking =', p[0])

    print('** SOMEONE IS TALKING ***') if p[0] > 0.10 else print('no talking...')
