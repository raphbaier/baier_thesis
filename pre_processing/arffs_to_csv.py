# -----------------------------------------------------------
# Get a csv containing the training and test data for the earnings
# calls and the stock price changes. This will apply the chosen
# feature-engineering steps.
#
# -----------------------------------------------------------

import os
from datetime import datetime
from datetime import timedelta

import arff
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from file_names import FEATURES_DIRECTORY, TRANSCRIPTS_DIRECTORY

'''
Get the mfcc-feature band from a given arff file.
'''
def get_mfcc_from_arff(file):
    found = True
    try:
        with open(file) as arff_opened:
            arff_file = arff.load(arff_opened)
        arff_opened.close()
    except:
        print(file)
        found = False
        # with open(
        #        "/home/raphael/masterSpassGit/model/cnn/input_arff/female/disgust/_03-01-07-01-01-01-02.arff") as file2:
        #    arff_file = arff.load(file2)
        # file2.close()
    if found:
        counter = 0
        derived = True
        mfcc_list = []

        last_counter = "0"
        new_mfcc = []
        for attribute in arff_file['attributes']:
            attribute_found = False
            if 'mfcc_sma' in attribute[0]:
                if not derived:
                    if not "sma_de" in attribute[0]:
                        attribute_found = True
                else:
                    attribute_found = True

            if attribute_found:
                if len(arff_file['data']) > 0:
                    new_mfcc.append(arff_file['data'][0][counter])
            counter += 1

        new_mfcc = np.array(new_mfcc)

        if len(new_mfcc) == 0:
            new_mfcc = np.zeros((30, 21))
        else:
            new_mfcc = new_mfcc.reshape((30, 21))
    else:
        # if file is not found use a file with zeros.
        mfcc_list = []
        new_mfcc = np.zeros((30, 21))
    mfcc_list.append(new_mfcc)

    return mfcc_list


'''
Save a csv file containing training or test paths and the labels.
'''
def get_csv(train_data, train_proportion, onlyQAndA, onlyExtremes,
            onlyOutliers, normalizeSpeakers, consider_earnings_numbers):
    info_position_in_feature_path = len(FEATURES_DIRECTORY.split("/")) - 1

    DAYS_BEFORE = 1
    DAYS_AFTER = 1

    talk_amount = 0
    for file in os.listdir(FEATURES_DIRECTORY):
        talk_amount += 1

    train_index = int(train_proportion * talk_amount)
    stock_growth_list_m = []

    paths_list_m = []

    # proportion of highest and lowest stocks we want to keep --- 0.5 is the maximum
    extreme_proportion = 0.3
    counter = 0
    for file in os.listdir(FEATURES_DIRECTORY):
        if (train_data and counter < train_index) or (not train_data and counter > train_index):
            stock_name = file.split("-")[0]
            # only stop training when the last ticker symbol changes: Else, the same company will
            # appear in training and testing.

            print(file)

            # find out the date of the earnings call
            with open(TRANSCRIPTS_DIRECTORY + file + ".txt") as transcript:
                date_name = transcript.readlines()[2]
            transcript.close()
            print(stock_name)
            print(date_name)
            stock_growth = 0
            if "Earnings" in date_name:
                date = date_name.split(" ")[1]
                if date.count("-") == 2:
                    try:
                        ticker = yf.Ticker(stock_name)
                        if ticker.info['quoteType'] == "EQUITY":
                            datetime_object = datetime.strptime(date, '%m-%d-%y')
                            start_date = datetime_object - timedelta(days=DAYS_BEFORE)
                            end_date = datetime_object + timedelta(days=DAYS_AFTER)
                            stock_data = yf.download(stock_name, start_date.date(), end_date.date())
                            first_course = stock_data.Close[0]
                            second_course = stock_data.Close[-1]
                            stock_growth = (second_course / first_course) - 1
                            print(stock_growth)

                            if consider_earnings_numbers != 0:
                                with open(FEATURES_DIRECTORY + file + "/" + "earnings_numbers.txt",
                                          "r") as earnings_numbers:
                                    deviations = earnings_numbers.read().split(" ")
                                earnings_numbers.close()
                                print(deviations)
                                # the deviation from the "normal" stock course
                                if '' in deviations:
                                    deviations.remove('')
                                try:
                                    stock_growth = stock_growth - float(deviations[consider_earnings_numbers - 1])
                                except:
                                    print("Unknown deviation.")


                    except KeyError:
                        print("Couldn't find symbol.")

            # start from here for the q&a session
            qanda = 0
            try:
                with open(FEATURES_DIRECTORY + file + "/qAndACount.txt", "r") as qaFile:
                    qanda = int(qaFile.read())
                qaFile.close()

                for arff_file in os.listdir(FEATURES_DIRECTORY + file + "/arff"):
                    infos = arff_file.split("_")
                    if int(infos[0]) > qanda or not onlyQAndA:
                        if infos[1] == "m":
                            stock_growth_list_m.append(stock_growth)
                            paths_list_m.append(FEATURES_DIRECTORY + file + "/arff/" + arff_file)
            except:
                print("file not found. Probably no Q&A section found or no "
                      "earnings numbers for earnings consideration.")
        counter += 1

    ### only the highest and lowest stock courses
    if not onlyExtremes:
        extreme_proportion = 0.5

    stock_growth_list_m = np.array(stock_growth_list_m)
    paths_list_m = np.array(paths_list_m)

    ordering_m = stock_growth_list_m.argsort()
    paths_list_m = paths_list_m[ordering_m]

    proportion_m = int(extreme_proportion * len(paths_list_m))

    paths_list_m_lower = paths_list_m[:proportion_m]
    paths_list_m_higher = paths_list_m[-proportion_m:]

    lower_tickers = []
    higher_tickers = []
    for path in paths_list_m_lower:
        lower_tickers.append(path.split("/")[info_position_in_feature_path])
    for path in paths_list_m_higher:
        higher_tickers.append(path.split("/")[info_position_in_feature_path])
    lower_tickers = np.array(lower_tickers)
    higher_tickers = np.array(higher_tickers)
    ordering_lower = lower_tickers.argsort()
    ordering_higher = higher_tickers.argsort()
    paths_list_m_lower = paths_list_m_lower[ordering_lower]
    paths_list_m_higher = paths_list_m_higher[ordering_higher]

    ### keep only the outliers
    def keep_outliers(paths):
        new_paths = []
        paths_list = []
        current_path = paths[0].split("/")[info_position_in_feature_path]
        for path in paths:
            path_info = path.split("/")

            if path_info[info_position_in_feature_path] == current_path:
                new_paths.append(path)
            else:
                current_path = path_info[info_position_in_feature_path]
                new_mfccs = []
                paths_to_check = []
                for path_to_look_at in new_paths:
                    mfcc_to_look_at = get_mfcc_from_arff(path_to_look_at)
                    if mfcc_to_look_at != None:
                        new_mfccs.append(mfcc_to_look_at)
                        paths_to_check.append(path_to_look_at)

                new_paths = paths_to_check

                arr = np.array(new_mfccs)
                arr = np.reshape(arr, (arr.shape[0], 630))
                df = pd.DataFrame(arr)

                no_outliers = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
                to_remove_from_paths = []
                for index, row in no_outliers.iterrows():
                    print(index)
                    to_remove_from_paths.append(index)

                counter = 0
                paths_to_add = []
                for new_path in new_paths:
                    if counter not in to_remove_from_paths:
                        paths_to_add.append(new_path)
                    counter += 1

                paths_list.append(paths_to_add)

                new_paths = []

        return paths_list

    if onlyOutliers:
        paths_list_m_lower = keep_outliers(paths_list_m_lower)
        paths_list_m_higher = keep_outliers(paths_list_m_higher)
    path_m = []
    emotion_m = []

    speech_infos = []
    speech_info_emotion = []

    def append_to_lists(path_list, path, emotion_list, emotion, speech_info_list, speech_info_emotion_list):
        path_list.append(path)
        emotion_list.append(emotion)
        if path.split("/")[info_position_in_feature_path] not in speech_info_list:
            speech_info_list.append(path.split("/")[info_position_in_feature_path])
            speech_info_emotion_list.append(emotion)
        return path_list, emotion_list, speech_info_list, speech_info_emotion_list

    for paths_per_earning in paths_list_m_lower:
        if onlyOutliers:
            for path in paths_per_earning:
                path_m, emotion_m, speech_infos, speech_info_emotion = append_to_lists(path_m, path,
                                                                                       emotion_m, "NEGATIVE",
                                                                                       speech_infos,
                                                                                       speech_info_emotion)
        else:
            path_m, emotion_m, speech_infos, speech_info_emotion = \
                append_to_lists(path_m, paths_per_earning, emotion_m, "NEGATIVE", speech_infos, speech_info_emotion)

    for paths_per_earning in paths_list_m_higher:
        if onlyOutliers:
            for path in paths_per_earning:
                path_m, emotion_m, speech_infos, speech_info_emotion = \
                    append_to_lists(path_m, path,
                                    emotion_m, "POSITIVE",
                                    speech_infos,
                                    speech_info_emotion)
        else:
            path_m, emotion_m, speech_infos, speech_info_emotion = \
                append_to_lists(path_m, paths_per_earning, emotion_m,
                                "POSITIVE", speech_infos, speech_info_emotion)

    negative_sentence_counter = 0
    positive_sentence_counter = 0

    if normalizeSpeakers:
        ref = pd.read_csv("speakersAndTheirSpeeches.csv")  # 354895
        print(ref.head())
        counter = 0
        most_sentence_counter = 0
        most_sentence_index = 0
        most_talks_counter = 0
        most_talks_index = 0

        most_emotion_counter = 0
        positive_emotion = 0
        negative_emotion = 0
        most_emotion_index = 0

        emotion_index_list = []
        positive_list = []
        negative_list = []

        sentences_to_remove = []

        for name in ref.participants:

            if "Patricia" in name:
                print(name)
            if "Patricia" not in name:
                sentence_counter = 0
                current_speech_infos = []
                if str(type(ref.speeches[counter])) != "<class 'float'>":
                    positive_sentences = []
                    negative_sentences = []

                    positive_emotion_counter = 0
                    negative_emotion_counter = 0
                    for speech_sentence in ref.speeches[counter].split(", "):
                        sentence_counter += 1
                        speech_info = speech_sentence.split("/")[info_position_in_feature_path]
                        if speech_info not in current_speech_infos:
                            current_speech_infos.append(speech_info)
                        emotion = "NONE"
                        if speech_info in speech_infos:
                            emotion = speech_info_emotion[speech_infos.index(speech_info)]
                        if emotion == "POSITIVE":
                            positive_emotion_counter += 1
                            positive_sentences.append(speech_sentence)
                        if emotion == "NEGATIVE":
                            negative_emotion_counter += 1
                            negative_sentences.append(speech_sentence)
                    emotion_counter = abs(positive_emotion_counter - negative_emotion_counter)

                    if emotion_counter > most_emotion_counter:
                        most_emotion_counter = emotion_counter
                        positive_emotion = positive_emotion_counter
                        negative_emotion = negative_emotion_counter
                        most_emotion_index = counter

                    emotion_index_list.append(counter)
                    positive_list.append(positive_emotion_counter)
                    negative_list.append(negative_emotion_counter)

                    if positive_emotion_counter - negative_emotion_counter > 0:
                        amount_to_remove = positive_emotion_counter - negative_emotion_counter - 0
                        remove_counter = 0
                        for sentence in positive_sentences:
                            if remove_counter < amount_to_remove:
                                sentences_to_remove.append(sentence)
                            remove_counter += 1
                            negative_sentence_counter += 1
                    elif negative_emotion_counter - positive_emotion_counter > 0:
                        amount_to_remove = negative_emotion_counter - positive_emotion_counter - 0
                        remove_counter = 0
                        for sentence in negative_sentences:
                            if remove_counter < amount_to_remove:
                                sentences_to_remove.append(sentence)
                            remove_counter += 1
                            positive_sentence_counter += 1

                    talks_counter = len(current_speech_infos)
                    if talks_counter > most_talks_counter:
                        most_talks_counter = talks_counter
                        most_talks_index = counter

                    if sentence_counter > most_sentence_counter:
                        most_sentence_counter = sentence_counter
                        most_sentence_index = counter
            counter += 1
        print(counter)
        print("The one with the most sentences:")
        print(ref.participants[most_sentence_index])
        print(most_sentence_counter)
        print("The one with the most talks:")
        print(ref.participants[most_talks_index])
        print(most_talks_counter)
        print("The one with the absolutely most biased emotion:")
        print(ref.participants[most_emotion_index])
        print(most_emotion_counter)
        print(positive_emotion)
        print(negative_emotion)

    stock_arff_df = pd.DataFrame(emotion_m, columns=['labels'])
    stock_arff_df = pd.concat([stock_arff_df, pd.DataFrame(path_m, columns=['path'])], axis=1)
    stock_arff_df.labels.value_counts()

    print("OKOKOKOKOK")
    print(len(stock_arff_df))
    counter = 0
    if normalizeSpeakers:
        for sentence in sentences_to_remove:
            if counter % 10 == 0:
                print(counter)
                print(len(stock_arff_df))
            stock_arff_df = stock_arff_df[stock_arff_df.path != sentence]
    print(len(stock_arff_df))

    arff_string = ""
    if onlyExtremes:
        arff_string += "onlyExtremes"
    if onlyOutliers:
        arff_string += "onlyOutliers"
    if onlyQAndA:
        arff_string += "onlyQAndA"
    if consider_earnings_numbers != 0:
        arff_string += "consider" + str(consider_earnings_numbers)
    if normalizeSpeakers:
        arff_string += "speakerNormalized"
    if train_data:
        arff_string += "_training"
    if not train_data:
        arff_string += "_testing"

    stock_arff_df.head()
    stock_arff_df.to_csv("arff_data_m" + arff_string + ".csv", mode='w', index=False, header=False)
