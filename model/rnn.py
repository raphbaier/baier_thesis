# -----------------------------------------------------------
# The functions for the CNN.
# With help from https://www.tensorflow.org/guide/keras/rnn
# -----------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import re

import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.preprocessing import sequence

batch_size = 64  # bei 2000 54%...
# Each MNIST image batch is a tensor of shape (batch_size, 28, 28).
# Each input sequence will be of size (28, 28) (height is treated like time).
input_dim = 7

units = 64
output_size = 2  # labels are from 0 to 9
TRAIN_PROPORTION_RNN = 0.7


def _atoi(text):
    return int(text) if text.isdigit() else text

'''
alist.sort(key=_natural_keys) sorts in human order
http://nedbatchelder.com/blog/200712/human_sorting.html
(See Toothy's implementation in the comments)
'''
def _natural_keys(text):
    return [_atoi(c) for c in re.split(r'(\d+)', text)]


'''
Build the RNN model. Taken from the TensorFlow guide from
https://www.tensorflow.org/guide/keras/rnn
'''
def _build_model(allow_cudnn_kernel=True):
    # CuDNN is only available at the layer level, and not at the cell level.
    # This means `LSTM(units)` will use the CuDNN kernel,
    # while RNN(LSTMCell(units)) will run on non-CuDNN kernel.
    if allow_cudnn_kernel:
        # The LSTM layer with default options uses CuDNN.
        #more layers didn't show any improvement

        lstm_layer = tf.keras.layers.LSTM(units, input_shape=(None, input_dim))
    else:
        # Wrapping a LSTMCell in a RNN layer will not use CuDNN.
        lstm_layer = tf.keras.layers.RNN(
            tf.keras.layers.LSTMCell(units),
            input_shape=(None, input_dim))


        ###for bidirectional ###
    #    lstm_layer = tf.keras.layers.Bidirectional(
    #        tf.keras.layers.LSTMCell(units),
    #        input_shape=(None, input_dim))

    model = tf.keras.models.Sequential([
        lstm_layer,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(output_size, activation='softmax')]
    )
    return model

'''
Classifying stock price changed based on emotional input from
successive sentences.
'''
def classify_with_rnn(arff_file):
    model = _build_model(allow_cudnn_kernel=True)
    # x_test = x_train
    # y_test = y_train


    # reading data and pre-processing
    reader = csv.reader(open(arff_file), delimiter=",")
    sorted_list = sorted(reader, key=lambda row: _natural_keys(row[1]), reverse=False)
    # for row in sorted_list:
    #    print(row)

    positive_X = []
    positive_Y = []
    negative_X = []
    negative_Y = []
    new_x = []

    start_new_x = True
    currently_positive = False
    previous_sentence_number = -1
    previous_speaker_number = -1

    def add_emotions_to_x(row, x):
        x.append(
            [float(row[2]), float(row[3]), float(row[4]), float(row[5]),
             float(row[6]), float(row[7]), float(row[8])])
        return x

    for row in sorted_list[:-1]:
        # print(row)
        sentence_id = row[1].split("/")[8].split("_")
        sentence_number = int(sentence_id[0])
        speaker_number = int(sentence_id[2][:-5])

        if start_new_x:
            new_x = add_emotions_to_x(row, new_x)
            start_new_x = False
            previous_sentence_number = sentence_number
            previous_speaker_number = speaker_number
            if row[0] == "POSITIVE":
                positive_Y.append(1)
                currently_positive = True
            if row[0] == "NEGATIVE":
                negative_Y.append(0)
                currently_positive = False
        else:
            # next sentence in speech of the same talker
            if previous_sentence_number + 1 == sentence_number and previous_speaker_number == speaker_number:
                # new_x.append([row[2], row[3], row[4], row[5], row[6], row[7], row[8]])
                new_x = add_emotions_to_x(row, new_x)
            else:
                if currently_positive:
                    positive_X.append(new_x)
                else:
                    negative_X.append(new_x)
                new_x = []
                new_x = add_emotions_to_x(row, new_x)

                if row[0] == "POSITIVE":
                    positive_Y.append(1)
                    currently_positive = True
                if row[0] == "NEGATIVE":
                    negative_Y.append(0)
                    currently_positive = False
        if row == sorted_list[-2]:
            if currently_positive:
                positive_X.append(new_x)
            else:
                negative_X.append(new_x)
        previous_sentence_number = sentence_number
        previous_speaker_number = speaker_number

    index = int(len(positive_X) * TRAIN_PROPORTION_RNN)
    positive_X_train = positive_X[:index]
    positive_Y_train = positive_Y[:index]
    negative_X_train = negative_X[:index]
    negative_Y_train = negative_Y[:index]

    positive_X_test = positive_X[index:]
    positive_Y_test = positive_Y[index:]
    negative_X_test = negative_X[index:]
    negative_Y_test = negative_Y[index:]

    X_train = positive_X_train + negative_X_train
    Y_train = positive_Y_train + negative_Y_train
    X_test = positive_X_test + negative_X_test
    Y_test = positive_Y_test + negative_Y_test

    maxlen = 300
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    print(X_train.shape)
    print(np.array(Y_train).shape)

    def unison_shuffled_copies(a, b):
        indices = np.arange(a.shape[0])
        np.random.shuffle(indices)

        a = a[indices]
        b = b[indices]
        return a, b

    X_train, Y_train = unison_shuffled_copies(X_train, np.array(Y_train))

    sgd = optimizers.SGD(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              validation_data=(X_test, Y_test),
              batch_size=batch_size,
              epochs=10)

    predicted_label = model.predict(X_test)
    true_label = Y_test
    pred_label = []
    for label in predicted_label:
        if label[0] > label[1]:
            pred_label.append(0)
        else:
            pred_label.append(1)

    print("The confusion matrix:")
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, [pred_label, true_label], 1)
    print(cm)
