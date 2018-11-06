import requests
import json

import numpy as np


url = 'http://127.0.0.1:5000/predict'
files = {'audio': open('data/test_set/test_with_dialog_00105.wav', 'rb')}

r = requests.post(url, files=files)
#print(r.text)

result_json = json.loads(r.text)
if result_json['success']:
    # create a numpy array from json
    p = np.array(result_json['predictions'])

    # transform probabilities into 0 and 1 with a threshold of 0.60
    preds = p >= 0.60
    preds = preds.astype(np.int)

    print(preds)
    print('mean prediction =', preds.mean())

    print('** SOMEONE IS TALKING ***') if preds.mean() > 0.20 else print('no talking...')
