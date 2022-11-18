import os
from pydub import AudioSegment
import speech_recognition as sr
from pydub.silence import split_on_silence


r = sr.Recognizer()


def get_large_audio_transcription():
    sound = AudioSegment.from_wav('output.wav')

    chunks = split_on_silence(sound,
                              min_silence_len=500,
                              silence_thresh=sound.dBFS - 14,
                              keep_silence=500,
                              )
    folder_name = "audio-chunks"
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""

    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")

        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print(str(e))
            else:
                text = f"{text.capitalize()}. "
                # print(text)
                whole_text += text

    text_file = open("output.txt", "w")
    text_file.write(whole_text)
    text_file.close()

    return whole_text


path = "output.wav"

# print(get_large_audio_transcription(path))


