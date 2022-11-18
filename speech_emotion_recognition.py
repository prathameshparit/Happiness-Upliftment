import librosa
import numpy as np
import pandas as pd
import IPython.display as ipd
from keras.models import model_from_json


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("models/Speech_Emotion_Recognition_Model.h5")
demo_audio_path = 'output.wav'


def get_audio_features(audio_path, sampling_rate):
    X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=sampling_rate * 2, offset=0.5)
    sample_rate = np.array(sample_rate)

    y_harmonic, y_percussive = librosa.effects.hpss(X)
    pitches, magnitudes = librosa.core.pitch.piptrack(y=X, sr=sample_rate)

    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=1)

    pitches = np.trim_zeros(np.mean(pitches, axis=1))[:20]

    magnitudes = np.trim_zeros(np.mean(magnitudes, axis=1))[:20]

    C = np.mean(librosa.feature.chroma_cqt(y=y_harmonic, sr=sampling_rate), axis=1)

    return [mfccs, pitches, magnitudes, C]


def get_features_dataframe(dataframe, sampling_rate):
    labels = pd.DataFrame(dataframe['label'])

    features = pd.DataFrame(columns=['mfcc', 'pitches', 'magnitudes', 'C'])
    for index, audio_path in enumerate(dataframe['path']):
        features.loc[index] = get_audio_features(audio_path, sampling_rate)

    mfcc = features.mfcc.apply(pd.Series)
    pit = features.pitches.apply(pd.Series)
    mag = features.magnitudes.apply(pd.Series)
    C = features.C.apply(pd.Series)

    combined_features = pd.concat([mfcc, pit, mag, C], axis=1, ignore_index=True)

    return combined_features, labels


def speech_emotion_recognition():
    ipd.Audio('output.wav')

    demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(demo_audio_path, 20000)

    mfcc = pd.Series(demo_mfcc)
    pit = pd.Series(demo_pitch)
    mag = pd.Series(demo_mag)
    C = pd.Series(demo_chrom)
    demo_audio_features = pd.concat([mfcc, pit, mag, C], ignore_index=True)

    demo_audio_features = np.expand_dims(demo_audio_features, axis=0)
    demo_audio_features = np.expand_dims(demo_audio_features, axis=2)

    livepreds = loaded_model.predict(demo_audio_features,
                                     batch_size=32,
                                     verbose=1)
    emotions = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    index = livepreds.argmax(axis=1).item()

    return emotions[index]

# print(speech_emotion_recognition(demo_audio_path))
