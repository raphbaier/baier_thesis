# -----------------------------------------------------------
# The functions for the SVM.
# -----------------------------------------------------------

import random

import numpy as np
import pandas as pd
from sklearn import svm, linear_model

TRAIN_PROPORTION_SVM = 0.7

'''
Based on emotional input (vector with size 7), classifying for
stock price change.
'''
def classify_with_svm(arff_file, mode):
    ref = pd.read_csv(arff_file)
    train_index = int(len(ref["angry"])*TRAIN_PROPORTION_SVM)
    print(train_index)

    #c = np.concatenate(a, b)
    X = np.column_stack((ref["angry"], ref["fear"], ref["happy"], ref["sad"],
                         ref["surprise"], ref["neutral"], ref["disgust"]))
    print(len(X))
    print(len(ref["angry"]))
    Y = ref["labels"]
    index = int(len(X)/2)
    X_neg = X[:index]
    X_pos = X[index:]
    Y_neg = Y[:index]
    Y_pos = Y[index:]


    c = list(zip(X_neg, Y_neg))
    random.shuffle(c)
    X_neg, Y_neg = zip(*c)
    c = list(zip(X_pos, Y_pos))
    random.shuffle(c)
    X_pos, Y_pos = zip(*c)

    index = int(index*TRAIN_PROPORTION_SVM)
    X_neg_train = X_neg[:index]
    X_pos_train = X_pos[:index]
    Y_neg_train = Y_neg[:index]
    Y_pos_train = Y_pos[:index]

    X_neg_test = X_neg[index:]
    X_pos_test = X_pos[index:]
    Y_neg_test = Y_neg[index:]
    Y_pos_test = Y_pos[index:]

    X_train = np.concatenate((X_neg_train, X_pos_train))
    X_test = np.concatenate((X_neg_test, X_pos_test))
    Y_train = np.concatenate((Y_neg_train, Y_pos_train))
    Y_test = np.concatenate((Y_neg_test, Y_pos_test))

    clf = linear_model.SGDClassifier(tol=1e-3)
    if mode == "svc":
        clf = svm.SVC()
    clf.fit(X_train, Y_train)

    counter = 0

    cm = np.zeros((2, 2), dtype=int)

    predicted_label = []
    true_label = []

    for test_x in X_test:
        prediction = clf.predict([test_x])
        true_class = 0
        predicted_class = 0
        if prediction == "POSITIVE":
            predicted_class = 1
        if Y_test[counter] == "POSITIVE":
            true_class = 1
        predicted_label.append(predicted_class)
        true_label.append(true_class)

        counter += 1

    np.add.at(cm, [predicted_label, true_label], 1)
    print("Confusion Matrix of the output: ")
    print(cm)


'''
Based on predicted stock price change (vector of size 2 (can also be converted to size 1
as sum is always 1)), predict the emotion.
'''
def classify_emotions_from_earnings(earnings_labels, y, use_binary_emotions):
    negative_emotions = ["male_angry", "male_disgust", "male_fear", "male_sad"]
    #distinguish between positive and negative emotions
    binary_emotions = []
    positive_counter = 0
    negative_counter = 0
    if use_binary_emotions:
        for emotion in y:
            if emotion in negative_emotions:
                binary_emotions.append("negative_emotion")
                positive_counter += 1
            else:
                binary_emotions.append("positive_emotion")
                negative_counter += 1

        print("counters")
        print(positive_counter)
        print(negative_counter)

        y = binary_emotions

    train_index = int(len(earnings_labels) * TRAIN_PROPORTION_SVM)
    earnings_labels_train = earnings_labels[:train_index]
    earnings_labels_test = earnings_labels[train_index:]
    y_train = y[:train_index]
    y_test = y[train_index:]

    clf = svm.SVC()
    clf.fit(earnings_labels_train, y_train)

    counter = 0
    true_counter = 0
    true_predicts = []
    for test in earnings_labels_test:
        prediction = clf.predict([test])
        if prediction == y_test[counter]:
            true_counter += 1
            true_predicts.append(prediction)

        counter += 1
    acc = true_counter / counter
    print(acc)
    print(true_counter)
    print(counter)