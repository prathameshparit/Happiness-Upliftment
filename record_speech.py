import pyaudio
import cv2
import wave
from speech_emotion_recognition import speech_emotion_recognition
from speech_to_text import get_large_audio_transcription
from text_emotion_recognition import predict
import speech_recognition as sr
print(sr.Microphone.list_microphone_names())
chunk = 1024
sample_format = pyaudio.paInt16
channels = 2
fs = 44100
seconds = 3
filename = "output.wav"

p = pyaudio.PyAudio()


def record_speech():
    print('Recording')

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []


    # Store data in chunks for 3 seconds
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)


    stream.stop_stream()
    stream.close()

    p.terminate()

    print('Finished recording')

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    return


path = "output.wav"
# record_speech()
# print(f"Speech Emotion: {speech_emotion_recognition()}")
# print(f"Text: {get_large_audio_transcription()}")
# print(f"Text Emotion: {predict()}")

