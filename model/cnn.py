# -----------------------------------------------------------
# The functions for the CNN.
#
# -----------------------------------------------------------


import csv

import numpy as np
import pandas as pd
from keras import losses, models, optimizers
from keras.activations import softmax
from keras.layers import (Convolution2D, BatchNormalization, Flatten, Dropout,
                          MaxPool2D, Activation, Input, Dense)
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from model.svm import classify_emotions_from_earnings
from pre_processing.arffs_to_csv import get_mfcc_from_arff

TRAINING_PROPORTION = 0.75

'''
Return two arrays shuffled in the same order
'''


def _unison_shuffled_copies(a, b):
    indices = np.arange(a.shape[0])
    np.random.shuffle(indices)

    a = a[indices]
    b = b[indices]
    return a, b


'''
Extracting the MFCC features as different bands.
With help from https://www.kaggle.com/ejlok1/audio-emotion-part-6-2d-cnn-66-accuracy
'''
def _prepare_data(df, n, mfcc_for_emotions):
    X = np.empty(shape=(df.shape[0], n, 21, 1))

    counter = 0
    for fname in tqdm(df.path):
        if not mfcc_for_emotions:
            mfccc = get_mfcc_from_arff(fname)
            mfccc = np.expand_dims(mfccc, axis=-1)
            X[counter,] = mfccc

        elif mfcc_for_emotions:
            file = "data/emotional_speech_sets/input/"  # _telephone
            if "happy" in df.labels[counter]:
                file += "happy/"
            if "sad" in df.labels[counter]:
                file += "sad/"
            if "fear" in df.labels[counter]:
                file += "fear/"
            if "disgust" in df.labels[counter]:
                file += "disgust/"
            if "neutral" in df.labels[counter]:
                file += "neutral/"
            if "surprise" in df.labels[counter]:
                file += "surprise/"
            if "angry" in df.labels[counter]:
                file += "angry/"
            file += "_"
            file += fname.split("/")[-1].replace(".wav", ".arff")
            mfccc = get_mfcc_from_arff(file)
            mfccc = np.expand_dims(mfccc, axis=-1)
            X[counter,] = mfccc
        counter += 1

    return X

'''
Build the CNN.
With help from https://www.kaggle.com/ejlok1/audio-emotion-part-6-2d-cnn-66-accuracy
'''
def _get_2d_conv_model(n, nclass):
    ''' Create a standard deep 2D convolutional neural network'''
    inp = Input(shape=(n, 21, 1))

    x = Convolution2D(32, (4, 10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)
    x = Dropout(rate=0.2)(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = Dropout(rate=0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(rate=0.2)(x)

    out = Dense(nclass, activation=softmax)(x)
    model = models.Model(inputs=inp, outputs=out)

    opt = optimizers.Adam(0.001)  # 0.001
    model.compile(optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

'''
Get input and labels for a csv file.
'''
def _getXAndY(csv_file):
    ref = pd.read_csv(csv_file)
    ref.columns = ['labels', 'path']
    n_mfcc = 30
    print("PREPARING...")
    mfcc = _prepare_data(ref, n_mfcc, False)
    print("PREPARED.")
    counter = 0
    last_negative = counter
    for label in ref.labels:
        if label == "NEGATIVE":
            last_negative = counter
        counter += 1
    X_pos, X_neg = np.split(mfcc, [last_negative])
    Y_pos, Y_neg = np.split(ref.labels, [last_negative])
    X = np.concatenate((X_pos, X_neg))
    Y = np.concatenate((Y_pos, Y_neg))
    return X, Y

'''
Train the model on earnings calls and stock price labels.
No test data used for validation, to make the code run faster.
'''
def train_model(saved_name, csv_file):
    print(saved_name)
    print(csv_file)
    n_mfcc = 30
    X_train, Y_train = _getXAndY(csv_file)

    X_test, y_test = _getXAndY("arff_data_monlyExtremesonlyOutliersonlyQAndAspeakerNormalized_testing.csv")

    lb = LabelEncoder()
    print(lb.fit_transform(Y_train))
    y_train = np_utils.to_categorical(lb.fit_transform(Y_train))

    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    # shuffle training data
    X_train, y_train = _unison_shuffled_copies(X_train, y_train)

    # build model
    model = _get_2d_conv_model(n_mfcc, 2)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, verbose=2, epochs=70)
    model.save("saved_models/" + saved_name + "70epochs.h5")

    ####save another json
    #model_json = model.to_json()
    #with open("saved_models/" + saved_name + "70epochs.json", "w") as json_file:
    #    json_file.write(model_json)


'''
Test the model trained on earnings calls for the stock price.
'''
def test_model(load_name, csv_file):
    print(load_name)

    X_test, Y_test = _getXAndY(csv_file)
    lb = LabelEncoder()
    print(lb.fit_transform(Y_test))
    Y_test = np_utils.to_categorical(lb.fit_transform(Y_test))
    print(Y_test)

    predicted_label = []
    true_label = []

    json_file = open("saved_models/" + "cnn_model_4hidden" + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("saved_models/" + load_name + ".h5")
    print("Loaded model from disk")
    for label in model.predict(X_test):
        if label[1] > label[0]:  # positive
            predicted_label.append(1)
        else:
            predicted_label.append(0)
    for label in Y_test:  # y_test
        if label[1] > label[0]:
            true_label.append(1)  # 1 means positive
        else:
            true_label.append(0)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, [predicted_label, true_label], 1)
    print("The Confusion Matrix of the Result:")
    print(cm)

'''
Test the model based on its ability to predict emotions.
'''
def test_model_emotion(load_name, use_binary_emotions):
    # loading json and model architecture
    json_file = open("saved_models/" + "cnn_model_4hidden" + ".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    model = models.model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights("saved_models/" + load_name + ".h5")
    print("Loaded model from disk")

    ref = pd.read_csv("data/emotional_speech_sets/Data_path_male.csv")
    ref.columns = ['index', 'labels', 'source', 'path']
    ref.head()
    n_mfcc = 30

    print("PREPARING...")
    mfcc = _prepare_data(ref, n_mfcc, True)
    print("PREPARED.")

    lb = LabelEncoder()
    X = mfcc

    print(lb.fit_transform(ref.labels))
    Y = np_utils.to_categorical(lb.fit_transform(ref.labels))

    counter = 0
    emotion_labels = []
    leave_out_values = []
    emotions_to_leave_out = ["male_surprise"]  # disgust und sad
    if use_binary_emotions:
        emotions_to_leave_out.append("male_disgust")
        emotions_to_leave_out.append("male_sad")

    for y in ref.labels:
        if y not in emotions_to_leave_out:  # and y != "male_neutral" and y != "male_sad":
            emotion_labels.append(y)
        else:
            leave_out_values.append(counter)
        counter += 1

    counter = 0
    positive_emotions = np.zeros(len(Y[0]))
    negative_emotions = np.zeros(len(Y[0]))

    earnings_labels = []

    for label in model.predict(X):

        if label[1] - 0.3 > label[0]:  # positive; maybe some threshold?
            positive_emotions[np.argmax(Y[counter])] += 1
        else:
            negative_emotions[np.argmax(Y[counter])] += 1
        if counter not in leave_out_values:
            earnings_labels.append(label)
        counter += 1
    print("Emotions as a one-hot vector for [angry, disgust, fear, happy, neutral, sad, surprise]")
    print("POSITIVE EMOTIONS")
    print(positive_emotions)
    print("NEGATIVE EMOTIONS")
    print(negative_emotions)

    # see proportion of which emotions are classified as negative
    counter = 0
    all_emotions = []
    for emotion in negative_emotions:
        all_emotions.append(emotion / (emotion + positive_emotions[counter]))
        counter += 1
    print("Proportion of negative emotions:")
    print(all_emotions)

    earnings_labels, emotion_labels = _unison_shuffled_copies(np.array(earnings_labels), np.array(emotion_labels))
    classify_emotions_from_earnings(earnings_labels, emotion_labels, use_binary_emotions)


'''
This generates the neural network which trains on emotional data sets and can classify for normal emotions.
Then, a set of earnings calls is labeled for the perceived emotion.
'''
def train_model_emotion(csv_file_to_label):
    ref = pd.read_csv("data/emotional_speech_sets/Data_path_male.csv")
    ref.columns = ['index', 'labels', 'source', 'path']
    ref.head()
    n_mfcc = 30

    print("PREPARING...")
    mfcc = _prepare_data(ref, n_mfcc, True)
    print("PREPARED.")

    # Split between train and test
    X_train, X_test, y_train, y_test = train_test_split(mfcc
                                                        , ref.labels
                                                        , test_size=0.25
                                                        , shuffle=True
                                                        , random_state=42
                                                        )

    # one hot encode the target
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    # Build CNN model
    model = _get_2d_conv_model(n_mfcc, 7)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              batch_size=16, verbose=2, epochs=200)

    new_file = []
    new_file.append(["labels", "path", "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"])

    # Go through them 1 by 1
    with open(csv_file_to_label,
              "r") as reff:  # 3 ist QANDA, #4 is extremestocks
        for line in reff.readlines():
            # if counter < 100:
            reff_info = line.split(",")

            add_to_file = []
            add_to_file.append(reff_info[0])
            add_to_file.append(reff_info[1][:-1])
            feature = get_mfcc_from_arff(reff_info[1][:-1])
            feature = np.array(feature)
            try:
                feature = feature.reshape((1, 30, 21, 1))

                preds = model.predict(feature)
                for pred in preds[0]:
                    add_to_file.append(pred)
                new_file.append(add_to_file)
            except:
                print("Couldn't reshape array.")

    reff.close()

    # output speichern
    with open(csv_file_to_label[:-4] + "_labeled.csv", "w", newline="") as output:
        writer = csv.writer(output)
        writer.writerows(new_file)
    output.close()
    print("DONE.")
