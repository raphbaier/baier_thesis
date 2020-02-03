# -----------------------------------------------------------
# Pre-processing the conventional speech set data.
# With help Eu Jin Lok's python notebook in kaggle
# from https://www.kaggle.com/ejlok1/audio-emotion-part-1-explore-data
#
# -----------------------------------------------------------

import os
from scipy.signal import lfilter, butter
from scipy.io.wavfile import read, write
from numpy import array, int16


def butter_params(low_freq, high_freq, fs, order=5):
    nyq = 0.5 * fs
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, low_freq, high_freq, fs, order=5):
    b, a = butter_params(low_freq, high_freq, fs, order=order)
    y = lfilter(b, a, data)
    return y


CREMA_DIR = "original_sets/cremad/AudioWAV/"
RAVDESS_DIR = "original_sets/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
SURREY_DIR = "original_sets/surrey-audiovisual-expressed-emotion-savee/ALL/"
TORONTO_DIR = "original_sets/toronto-emotional-speech-set-tess/TESS_Toronto_emotional_speech_set_data/"

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def add_emotion(file, emotion):
    # add emotion
    if os.path.isfile(file):
        with open(file, "rb+") as new_arff:
            new_arff.seek(-4, os.SEEK_END)
            new_arff.truncate()
        new_arff.close()
        with open(file, "a") as new_arff:
            new_arff.write(str(EMOTIONS.index(emotion)))
        new_arff.close()


for file in os.listdir(CREMA_DIR):
    # looking up the female speakers
    female = [1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1018, 1020, 1021, 1024, 1025, 1028, 1029,
              1030, 1037, 1043, 1046, 1047, 1049,
              1052, 1053, 1054, 1055, 1056, 1058, 1060, 1061, 1063, 1072, 1073, 1074, 1075, 1076, 1078, 1079, 1082,
              1084, 1089, 1091]

    file_infos = file.split("_")
    file_infos = file_infos
    gender = "male"
    if int(file_infos[0]) in female:
        gender = "female"

    emotion = "unknown"
    if file_infos[2] == 'SAD':
        emotion = "sad"
    elif file_infos[2] == 'ANG':
        emotion = "angry"
    elif file_infos[2] == 'DIS':
        emotion = "disgust"
    elif file_infos[2] == 'FEA':
        emotion = "fear"
    elif file_infos[2] == 'HAP':
        emotion = "happy"
    elif file_infos[2] == 'NEU':
        emotion = "neutral"

    print(CREMA_DIR + file)

    try:
        fs, audio = read(CREMA_DIR + file)
        low_freq = 300.0
        high_freq = 3000.0
        filtered_signal = butter_bandpass_filter(audio, low_freq, high_freq, fs, order=6)
        fname = CREMA_DIR + file[:-4] + "_moded.wav"
        write(fname, fs, array(filtered_signal, dtype=int16))

        os.system("SMILExtract -C /home/raphael/opensmile-2.3.0/config/emobase2010.conf -I "
                  + CREMA_DIR + file[:-4] + "_moded.wav" + " -O "
                  + "input_arff_telephone/" + gender + "/" + emotion + "/" + "_" + file[:-4] + ".arff" + " -l 0")

        add_emotion("input_arff_telephone/" + gender + "/" + emotion + "/" + "_" + file[:-4] + ".arff", emotion)
    except:
        print("format not understood...")

for file in os.listdir(RAVDESS_DIR):
    for sound_file in os.listdir(RAVDESS_DIR + file):

        file_infos = sound_file[:-4].split("-")

        emotion = "unknown"

        if file_infos[2] == "01":
            emotion = "neutral"
        if file_infos[2] == "02":
            emotion = "neutral"
        if file_infos[2] == "03":
            emotion = "happy"
        if file_infos[2] == "04":
            emotion = "sad"
        if file_infos[2] == "05":
            emotion = "angry"
        if file_infos[2] == "06":
            emotion = "fear"
        if file_infos[2] == "07":
            emotion = "disgust"
        if file_infos[2] == "08":
            emotion = "surprise"

        # odd speaker number means male
        gender = "male"
        if int(file_infos[6]) % 2 == 0:
            gender = "female"

        print(RAVDESS_DIR + file + "/" + sound_file)

        try:
            fs, audio = read(RAVDESS_DIR + file + "/" + sound_file)
            low_freq = 300.0
            high_freq = 3000.0
            filtered_signal = butter_bandpass_filter(audio, low_freq, high_freq, fs, order=6)
            # fname = sys.argv[1].split('.wav')[0] + '_moded.wav'
            fname = RAVDESS_DIR + file + "/" + sound_file[:-4] + "_moded.wav"
            write(fname, fs, array(filtered_signal, dtype=int16))

            os.system("SMILExtract -C /home/raphael/opensmile-2.3.0/config/emobase2010.conf -I "
                      + RAVDESS_DIR + file + "/" + sound_file[:-4] + "_moded.wav" + " -O "
                      + "input_arff_telephone/" + gender + "/" + emotion + "/" + "_" + sound_file[
                                                                                       :-4] + ".arff" + " -l 0")

            add_emotion("input_arff_telephone/" + gender + "/" + emotion + "/" + "_" + sound_file[:-4] + ".arff",
                        emotion)
        except:
            print("format not understood...")

for file in os.listdir(SURREY_DIR):

    gender = "male"

    emotion = "unknown"
    if file[-8:-6] == '_a':
        emotion = "angry"
    elif file[-8:-6] == '_d':
        emotion = "disgust"
    elif file[-8:-6] == '_f':
        emotion = "fear"
    elif file[-8:-6] == '_h':
        emotion = "happy"
    elif file[-8:-6] == '_n':
        emotion = "neutral"
    elif file[-8:-6] == 'sa':
        emotion = "sad"
    elif file[-8:-6] == 'su':
        emotion = "surprise"

    print(SURREY_DIR + file)

    try:
        fs, audio = read(SURREY_DIR + file)
        low_freq = 300.0
        high_freq = 3000.0
        filtered_signal = butter_bandpass_filter(audio, low_freq, high_freq, fs, order=6)
        fname = SURREY_DIR + file[:-4] + "_moded.wav"
        write(fname, fs, array(filtered_signal, dtype=int16))

        os.system("SMILExtract -C /home/raphael/opensmile-2.3.0/config/emobase2010.conf -I "
                  + SURREY_DIR + file[:-4] + "_moded.wav" + " -O "
                  + "input_arff_telephone/" + gender + "/" + emotion + "/" + "_" + file[:-4] + ".arff" + " -l 0")
        add_emotion("input_arff_telephone/" + gender + "/" + emotion + "/" + "_" + file[:-4] + ".arff", emotion)
    except:
        print("format not understood...")

for file in os.listdir(TORONTO_DIR):
    for sound_file in os.listdir(TORONTO_DIR + file):

        gender = "female"
        emotion = "unknown"
        if file[4:].lower() == "angry":
            emotion = "angry"
        if file[4:].lower() == "disgust":
            emotion = "disgust"
        if file[4:].lower() == "fear":
            emotion = "fear"
        if file[4:].lower() == "happy":
            emotion = "happy"
        if file[4:].lower() == "neutral":
            emotion = "neutral"
        if file[4:].lower() == "pleasant_surprise":
            emotion = "surprise"
        if file[4:].lower() == "sad":
            emotion = "sad"

        if "moded" in sound_file:
            os.remove(TORONTO_DIR + file + "/" + sound_file)
            print("removed" + TORONTO_DIR + file + "/" + sound_file)

        else:
            print(TORONTO_DIR + file + "/" + sound_file)
            print(TORONTO_DIR + file + "/" + sound_file[:-4])

            try:
                fs, audio = read(TORONTO_DIR + file + "/" + sound_file)
                low_freq = 300.0
                high_freq = 3000.0
                filtered_signal = butter_bandpass_filter(audio, low_freq, high_freq, fs, order=6)
                fname = TORONTO_DIR + file + "/" + sound_file[:-4] + "_moded.wav"
                write(fname, fs, array(filtered_signal, dtype=int16))

                os.system("SMILExtract -C /home/raphael/opensmile-2.3.0/config/emobase2010.conf -I "
                          + TORONTO_DIR + file + "/" + sound_file[:-4] + "_moded.wav" + " -O "
                          + "input_arff_telephone/" + gender + "/" + emotion + "/" + "_" + sound_file[
                                                                                           :-4] + ".arff" + " -l 0")

                add_emotion("input_arff_telephone/" + gender + "/" + emotion + "/" + "_" + sound_file[:-4] + ".arff",
                            emotion)
            except:
                print("format not understood...")
