#For the data collection, as it doesn't directly dontribute to the output of the system,
#and to prevent from having to install needless packages, there's the extra script
#collect.py in data.collect_transcripts.

from pre_processing.align_talk_to_transcript import align
from pre_processing.get_speaker_from_alignment_sentence_boundaries import cut_to_mfcc, save_stock_information, remove_wavs
from pre_processing.arffs_to_csv import get_csv
from pre_processing.get_speeches_per_speaker import get_speeches_for_speakers
from model.cnn import train_model, test_model, train_model_emotion, test_model_emotion
from model.svm import classify_with_svm
from model.rnn import classify_with_rnn

train_proportion = 0.7

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--mode', metavar='string', required=True,
                        help='what script you want to run: preprocess-align, preprocess-get_speaker')
    args = parser.parse_args()

    if args.mode == "preprocess-align":
        align()
    elif args.mode == "preprocess-cut_mfcc":
        cut_to_mfcc()
        save_stock_information()
        remove_wavs()
    elif args.mode == "preprocess-get_speeches_per_speaker":
        get_speeches_for_speakers()
    elif args.mode == "preprocess-get_csv":
        train_data = True
        onlyQAndA = False
        onlyExtremes = False
        onlyOutliers = False
        normalizeSpeakers = False
        consider_earnings_numbers = 0
        get_csv(train_data, train_proportion, onlyQAndA, onlyExtremes, onlyOutliers, normalizeSpeakers, consider_earnings_numbers)
    elif args.mode == "train-normal":
        train_model("test_modelspeakerNormalized70epochs", "arff_data_mspeakerNormalized_training.csv")
    elif args.mode == "test-normal":
        test_model("test_modelspeakerNormalized70epochs", "arff_data_mspeakerNormalized_testing.csv")
    elif args.mode == "label-real_emotions":
        train_model_emotion("arff_data_m_training.csv")
    elif args.mode == "test-normal_to_emotion":
        binary_emotions = True
        test_model_emotion("test_modelspeakerNormalized70epochs", binary_emotions)
    elif args.mode == "test-svm_real_emo_on_earnings":
        #"svm" classifier, else sgd classifier
        classifier = "sgd"
        classify_with_svm("arff_data_m_labeled.csv", "sgd")
    elif args.mode == "test-rnn_real_emo_on_earnings":
        classify_with_rnn("arff_data_m_labeled")
    else:
        print("Invalid input for \'mode\': Choose either preprocess-align, preprocess-cut_mfcc")
