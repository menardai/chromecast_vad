import time
import pychromecast
import pyaudio
import wave
import requests
import json

import numpy as np

from pydub import AudioSegment


class VoiceActivityDetector(object):

    def __init__(self, vad_url='http://127.0.0.1:5000/predict', wav_output_filename='output.wav'):
        self.vad_url = vad_url
        self.wav_output_filename = wav_output_filename

        self.recorder = AudioRecording(2.05)

    def is_speech_detected(self, talking_threshold=0.20):
        """
        Record live audio from microphone and predict if someone is talking.

        Returns True/False or None is can't connect to web service
        """
        self.recorder.record_audio(self.wav_output_filename)

        self._preprocess_audio(self.wav_output_filename)

        p = self._get_prediction()
        if p is not None:
            print('  speech probability = {0:.2f}'.format(p))
            return p > talking_threshold

        return None

    def _get_prediction(self):
        """
        Predict if someone is talking in the saved audio file.

        Returns True/False or None is can't connect to web service
        """
        files = {'audio': open(self.wav_output_filename, 'rb')}
        try:
            # make a request to the voice activity detection web service
            response = requests.post(self.vad_url, files=files)
        except Exception:
            return None

        result_json = json.loads(response.text)
        if result_json['success']:
            p = np.array(result_json['predictions'])
            return p[0]

        return None

    def _preprocess_audio(self, filename, duration_ms=2000):
        """ Preprocess the audio to the correct format. """
        padding = AudioSegment.silent(duration=duration_ms)  # Trim or pad audio segment

        segment = AudioSegment.from_wav(filename)[:duration_ms]
        segment = padding.overlay(segment)
        segment = segment.set_frame_rate(44100)

        segment.export(filename, format='wav')


class Chromecast(object):

    def __init__(self):
        self.chromecasts = pychromecast.get_chromecasts()
        self.device_names = [cc.device.friendly_name for cc in self.chromecasts]
        self.user_selected_device_index = -1  # reset selected device
        self.cast = None

    def reconnect_to_latest_device(self):
        """
        Try to reconnect to the latest connected device based on content of <latest_connected_device.txt>.
        Return true if successful
        """
        try:
            with open('latest_connected_device.txt') as f:
                device_name = f.readlines()

            return self.connect(device_name[0])
        except Exception:
            return False

    def get_device_cast(self, device_name=None):
        """
        Returns device cast object for the specified device_name (if any) or the selected index.
        """
        if device_name is not None:
            cast = next(cc for cc in self.chromecasts if cc.device.friendly_name == device_name)
        else:
            cast = self.chromecasts[self.user_selected_device_index]

        cast.wait()    # Wait for cast device to be ready
        return cast

    def prompt_device_selection(self):
        print('Please select device:')
        for i, device_name in enumerate(self.device_names):
            print('  {} - {}'.format(i, device_name))

        user_input = input('Enter your choice [0, {}]:'.format(len(self.device_names)-1))
        try:
            input_index = int(user_input)
        except ValueError:
            print('invalid selection')
            input_index = -1

        if 0 <= input_index < len(self.device_names):
            self.user_selected_device_index = input_index
            print('Device selected:', self.device_names[self.user_selected_device_index])

    def connect(self, device_name=None):
        if self.user_selected_device_index != -1 or device_name is not None:
            self.cast = self.get_device_cast(device_name)

            device_name = self.cast.device.friendly_name
            print("Connected to:", device_name)

            with open("latest_connected_device.txt", "w") as text_file:
                text_file.write(device_name)
            return True
        else:
            print("Can't connect, no device selected")
            return False

    def get_volume_status(self):
        status = self.cast.status
        return {
            'level': status.volume_level,
            'is_muted': status.volume_muted,
            'is_active_input': status.is_active_input,
            'is_stand_by': status.is_stand_by
        }

    def is_music_playing(self):
        return self.cast.media_controller.is_playing

    def volume_up(self):
        new_volume_level = self.cast.volume_up()
        print('  volume down up {0:.2f}'.format(new_volume_level))
        return new_volume_level

    def volume_down(self):
        new_volume_level = self.cast.volume_down()
        print('  volume down to {0:.2f}'.format(new_volume_level))
        return new_volume_level

    def set_volume(self, volume_level):
        """ set the volume level: [0, 1] """
        print('  set volume to {0:.2f}'.format(volume_level))
        return self.cast.set_volume(volume_level)


class AudioRecording(object):
    
    def __init__(self, audio_record_seconds):
        self.audio_record_seconds = audio_record_seconds

        self.audio_chunk = 1024
        self.audio_format = pyaudio.paInt16
        self.audio_channels = 1
        self.audio_rate = 44100

    def record_audio(self, wav_output_filename):
        p = pyaudio.PyAudio()
        
        stream = p.open(format=self.audio_format,
                        channels=self.audio_channels,
                        rate=self.audio_rate,
                        input=True,
                        frames_per_buffer=self.audio_chunk)
        
        print("> recording")
        frames = []
        for i in range(0, int(self.audio_rate / self.audio_chunk * self.audio_record_seconds)):
            data = stream.read(self.audio_chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        # save on disk
        wf = wave.open(wav_output_filename, 'wb')
        wf.setnchannels(self.audio_channels)
        wf.setsampwidth(p.get_sample_size(self.audio_format))
        wf.setframerate(self.audio_rate)
        wf.writeframes(b''.join(frames))
        wf.close()


def test_chromecast():
    print('Looking for devices...\n')
    chromecast = Chromecast()
    reconnect_successful = chromecast.reconnect_to_latest_device()
    if not reconnect_successful:
        chromecast.prompt_device_selection()
        chromecast.connect()
    status = chromecast.get_volume_status()
    print('volume level = {}'.format(status['level']))
    print('is music playing = {}'.format(chromecast.is_music_playing()))
    time.sleep(2)
    chromecast.volume_up()
    time.sleep(3)
    chromecast.volume_down()


def test_is_someone_talking():
    vad = VoiceActivityDetector(vad_url='http://192.168.86.41:5000/predict')
    is_talking = vad.is_speech_detected()

    if is_talking is not None:
        if is_talking:
            print('*** SPEECH DETECTED ***')
    else:
        print("Can't connect to voice activity detection web service.")


def start_volume_control():
    # Voice Activity Detector
    vad = VoiceActivityDetector(vad_url='http://192.168.86.41:5000/predict')

    # Chromecast
    print('Looking for chromecast devices...\n')
    chromecast = Chromecast()
    reconnect_successful = chromecast.reconnect_to_latest_device()
    if not reconnect_successful:
        chromecast.prompt_device_selection()
        chromecast.connect()

    initial_volume = chromecast.get_volume_status()['level']
    print('initial volume = {0:.2f}'.format(initial_volume))

    volume_lowered = False
    steps_without_speech = 0

    while True:
        if chromecast.is_music_playing():
            # perform speech detection only if music is playing loud enough
            level = chromecast.get_volume_status()['level']

            if level >= 0.25 or volume_lowered:
                # music is playing, check for someone talking
                is_talking = vad.is_speech_detected()

                if is_talking is not None:
                    if is_talking:
                        print('  *** SPEECH DETECTED ***')
                        steps_without_speech = 0

                        if not volume_lowered:
                            # lower the volume
                            chromecast.set_volume(initial_volume * 0.65)
                            volume_lowered = True
                    else:
                        # no speech detected
                        steps_without_speech += 1
                        print('  steps_without_speech =', steps_without_speech)

                        if volume_lowered and steps_without_speech >= 3:
                            # volume has been previously lowered and
                            # no speech detected for at least 3 steps -> reset volume
                            volume_lowered = False

                            # raise volume
                            chromecast.set_volume(initial_volume)
                            time.sleep(1)
                else:
                    print("Can't connect to voice activity detection web service.")
            else:
                print('music playing softly: volume={0:.2f}'.format(level))
                time.sleep(2)
        else:
            # no music, so nothing to do except wait moment and check for music playing again
            print('no music playing')
            time.sleep(2)


if __name__ == "__main__":
    start_volume_control()
