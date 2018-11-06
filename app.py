import os
import werkzeug
import wave
import json
import flask

import numpy as np
import tensorflow as tf
from flask_restful import Resource, Api, reqparse

from vad_model import VadModel

'''
IMPORTANT NOTE:

    TensorFlow, Keras, Flask - Unable to run my keras model as web app via flask
    https://stackoverflow.com/questions/49018861/tensorflow-keras-flask-unable-to-run-my-keras-model-as-web-app-via-flask
'''

class VoiceActivityDetection(Resource):

    def post(self):
        try:
            parse = reqparse.RequestParser()
            parse.add_argument('audio', type=werkzeug.FileStorage, location='files')

            args = parse.parse_args()

            # read the stream of the audio file
            stream = args['audio'].stream
            wav_file = wave.open(stream, 'rb')
            data = wav_file.readframes(-1)
            data = np.fromstring(data, 'Int16')
            fs = wav_file.getframerate()
            wav_file.close()

            # ask the model to make a prediction
            graph = app.config['GRAPH']
            vad_model = app.config['VAD_MODEL']
            with graph.as_default():
                predictions = vad_model.predict(wav_filename=None, rate=fs, data=data)

            result = {
                'success': True,
                'model_version': vad_model.version,

                'source_data_length': len(data),
                'source_data_rate': fs,

                'predictions': predictions[0].reshape(len(predictions[0])).tolist()
            }
        except Exception:
            result = {'success': False}

        return flask.jsonify(result)

    # def post(self):
    #     ''' Save multi-part on disk and use the saved file to make the prediction. '''
    #     parse = reqparse.RequestParser()
    #     parse.add_argument('audio', type=werkzeug.FileStorage, location='files')
    #
    #     args = parse.parse_args()
    #     audio_file = args['audio']
    #     audio_file.save('audio_received.wav')
    #
    #     # ask the model to make a prediction
    #     predictions = self.vad_model.predict(wav_filename='audio_received.wav')
    #
    #     statinfo = os.stat('audio_received.wav')
    #
    #     return {
    #         'prediction length': len(predictions),
    #         'audio file size': statinfo.st_size
    #     }


def create_app(graph, vad_model):
    app = flask.Flask(__name__)
    app.config['GRAPH'] = graph
    app.config['VAD_MODEL'] = vad_model

    api = Api(app)
    api.add_resource(VoiceActivityDetection, '/predict')

    return app



if __name__ == '__main__':
    graph = tf.Graph()
    with graph.as_default():
        vad_model = VadModel('models/model_architecture.json', 'models/vad_26_10_2018_weights_7.h5')

    app = create_app(graph, vad_model)
    app.run(host='0.0.0.0')
