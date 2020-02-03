# -----------------------------------------------------------
# From mp3 files and the alignments, extract the single sentences.
# Those are converted to arff files.
#
# -----------------------------------------------------------


import os
import json
import shutil
from pydub import AudioSegment
import numpy as np
import Levenshtein
import gender_guesser.detector
from file_names import SOUND_DIRECTORY, TRANSCRIPTS_DIRECTORY, ALIGNMENT_DIRECTORY, FEATURES_DIRECTORY

directory = os.fsencode(SOUND_DIRECTORY)

counter = 1

#the possible titles of the list of names of company members
IMPORTANT_NAMES = ["Company Participants", "Executives", "Company Representatives", "Corporate Participants"]
#list of important names ends here.
UNIMPORTANT_NAMES = ["Analysts", "Conference Call Participants", "Analyst"]
#End searching for names at the last name, operator
OPERATOR_NAMES = ["Operator"]


'''
Get the names of the people whose speech we want to extract.
'''
def _get_important_names_from_transcript(transcript):
    found_important_start = False
    found_unimportant_start = False
    names_list_filled = False
    list_of_important_names = []
    list_of_unimportant_names = []

    for line in transcript.readlines():
        if not names_list_filled:
            if found_important_start:

                for name in UNIMPORTANT_NAMES:
                    if name in line:
                        found_unimportant_start = True
                        found_important_start = False

                if not found_unimportant_start:
                    if line != '\n':
                        name_to_add = line.split('-')[0]
                        if name_to_add[-1] == ' ':
                            name_to_add = name_to_add[:-1]

                        list_of_important_names.append(name_to_add)

            elif found_unimportant_start:
                for name in OPERATOR_NAMES:
                    if name in line:
                        names_list_filled = True
                if not names_list_filled:
                    if line != '\n':
                        name_to_add = line.split('-')[0]
                        if len(name_to_add) > 0:
                            if name_to_add[-1] == ' ':
                                name_to_add = name_to_add[:-1]
                            list_of_unimportant_names.append(name_to_add)

            else:
                for name in IMPORTANT_NAMES:
                    if name in line:
                        # we found people belonging to the company
                        found_important_start = True

    print(list_of_important_names)
    print(list_of_unimportant_names)
    return [list_of_important_names, list_of_unimportant_names]

'''
A class containing the names of the speakers and the start and end times of
the sentences they speak in the given mp3 file and transcript.
'''
class Sound_Data:
    def __init__(self, transcript_file):

        gender_detector = gender_guesser.detector.Detector()

        with open(TRANSCRIPTS_DIRECTORY + transcript_file + '.txt', 'r') as transcript:
            self.names = _get_important_names_from_transcript(transcript)
        with open(ALIGNMENT_DIRECTORY + transcript_file + '_alignment.json', 'r') as alignment:
            self.alignment = json.load(alignment)
        all_words_in_alignment = []

        transcript_started = False
        important_started = False


        self.important_times = []

        self.sentence_boundaries = []

        self.qanda = 0

        in_important_time = False
        important_time = []

        counter = 0

        transcript_words = self.alignment['transcript']

        # See where the end of a sentence is
        sentence_replacements = [".", "?", "!"]
        to_replace = ["|", ":", ",", ";", "_", "-", "(", ")", "=", "'", "    ", "   ", "  "]

        for symbol in sentence_replacements:
            transcript_words = transcript_words.replace(symbol, "SENTENCEEND")
        for symbol in to_replace:
            transcript_words = transcript_words.replace(symbol, " ")

        #print(transcript_words)
        transcript_words = transcript_words.replace("\n\n", "\n").replace("\n", " \n ")
        transcript_words = transcript_words.split(" ")
        print(transcript_words)

        position_in_transcript = 0
        window_size = 3

        #the speeches' speakers' genders
        self.speaker_information_list = []

        false_words_counter = 0

        #start time of a sentence
        start_time = 0

        qAndAStarted = False

        for word in self.alignment['words']:



            transcrib_word = word['word']

            #get current position in transcript
            position_in_transcript = min(len(transcript_words)-1, position_in_transcript+1)

            smallest_distance = Levenshtein.distance(transcript_words[position_in_transcript].replace("SENTENCEEND", ""), transcrib_word)
            start_pos = max(0, position_in_transcript - window_size)
            end_pos = min(position_in_transcript + window_size, len(transcript_words))
            new_position = 0
            current_position = start_pos
            for word in transcript_words[start_pos:end_pos]:
                current_distance = Levenshtein.distance(word.replace("SENTENCEEND", ""), transcrib_word)
                if current_distance <= smallest_distance:
                    smallest_distance = current_distance
                    new_position = current_position
                current_position += 1
            position_in_transcript = new_position

            if transcrib_word != transcript_words[position_in_transcript]:
                false_words_counter += 1
            else:
                false_words_counter = 0

            sentence_end = False
            if "SENTENCEEND" in transcript_words[position_in_transcript]:
                sentence_end = True

            if not qAndAStarted:
                if position_in_transcript + 2 < len(transcript_words):
                    for i in range(position_in_transcript-2,position_in_transcript+2):
                        if transcript_words[i] == "Question":
                            if transcript_words[i+1] == "and":
                                if transcript_words[i+2] == "Answer":
                                    if transcript_words[i+3] == "Session":
                                        if transcript_words[i + 4] == "\n":
                                            qAndAStarted = True

            if not transcript_started:
                if transcrib_word == 'Operator':
                    transcript_started = True
            else:
                all_words_in_alignment.append(word)

                # write the start and end time of a word into the word boundaries list
                if self.alignment['words'][counter]['case'] == "success":
                    sentence_time = []
                    sentence_time.append(start_time)

                    #end time of a sentence
                    end_time = self.alignment['words'][counter]['end']
                    sentence_time.append(end_time)

                    if sentence_end:
                        if qAndAStarted:
                            self.qanda = sentence_time[0]
                            qAndAStarted = False
                        self.sentence_boundaries.append(sentence_time)
                        start_time = self.alignment['words'][counter]['end']




                #look for important names, i.e. an important speaker starts to speak from here
                important_found = False

                current_name = self.names[0][0]

                for name in self.names[0]:
                    #print(name)
                    if transcrib_word.lower() == name.split(" ")[0].replace(".", "").lower():
                        name_counter = 0
                        important_found = True
                        current_name = name
                        for subname in name.split(" "):

                            name_position = min(len(self.alignment['words']) - 1, counter + name_counter)
                            if self.alignment['words'][name_position]['word'].lower() != subname.replace('.', '').lower():
                                important_found = False
                            name_counter += 1

                        next_paragraph_position = min(len(transcript_words)-1, position_in_transcript + name_counter)
                        before_paragraph_position = max(0, position_in_transcript - 1)

                        #if transcript_words[next_paragraph_position] != "\n" or transcript_words[before_paragraph_position] != "\n":
                        if transcript_words[before_paragraph_position] != "\n":
                            important_found = False



                if important_found:
                    #we found a new important name but are already in another important name
                    if in_important_time:
                        #end current time: go back to the latest timestamp
                        counter_timestamp = counter - 1
                        while self.alignment['words'][counter_timestamp]['case'] != "success":
                            counter_timestamp -= 1
                        important_time.append(self.alignment['words'][counter_timestamp]['end'])
                        if len(important_time) == 2:
                            self.important_times.append(important_time)
                        important_time = []
                    #we found a new important name and are not in another important name
                    #we start a new important time
                    #start current time: go forward to the newest timestamp
                    counter_timestamp = counter + 1
                    while self.alignment['words'][counter_timestamp]['case'] != "success" and counter_timestamp < len(self.alignment['words'])-1:
                        counter_timestamp += 1
                    try:


                        # We get the speaker's gender. If it's unknown, we assume it to be male
                        speaker_gender = gender_detector.get_gender(current_name.split(" ")[0])
                        speaker_gender = speaker_gender.replace("mostly_", "")
                        speaker_gender = speaker_gender.replace("unknown", "male")[0]

                        speaker_id = str(self.names[0].index(current_name))
                        speaker_information = speaker_gender + "_" + speaker_id
                        self.speaker_information_list.append(speaker_information)

                        important_time.append(self.alignment['words'][counter_timestamp]['start'])
                        in_important_time = True

                        # set start time of new sentence
                        success_counter = counter + 1
                        while success_counter < len(self.alignment['words']) and \
                                        self.alignment['words'][success_counter]['case'] != "success":
                            success_counter += 1
                        if success_counter < len(self.alignment['words']) and self.alignment['words'][success_counter][
                            'case'] != "success":
                            start_time = self.alignment['words'][success_counter]['start']

                    except Exception as e:
                        print(e)

                unimportant_found = False
                for name in self.names[1] + OPERATOR_NAMES:
                    if transcrib_word.lower() == name.split(" ")[0].replace(".", "").lower():
                        name_counter = 0
                        unimportant_found = True
                        for subname in name.split(" "):

                            name_position = min(len(self.alignment['words'])-1, counter+name_counter)
                            if self.alignment['words'][name_position]['word'].lower() != subname.replace('.', '').lower():
                                unimportant_found = False
                            name_counter += 1

                        next_paragraph_position = min(len(transcript_words) - 1, position_in_transcript + name_counter)
                        before_paragraph_position = max(0, position_in_transcript - 1)

                        #if transcript_words[next_paragraph_position] != "\n" or transcript_words[before_paragraph_position] != "\n":
                        if transcript_words[before_paragraph_position] != "\n":
                            unimportant_found = False

                if unimportant_found:
                    #we found an important name and are in an important name: end current time: go back to the latest timestamp
                    if in_important_time:
                        counter_timestamp = counter - 1
                        while self.alignment['words'][counter_timestamp]['case'] != "success":
                            counter_timestamp -= 1
                        important_time.append(self.alignment['words'][counter_timestamp]['end'])
                        if len(important_time) == 2:
                            self.important_times.append(important_time)
                        important_time = []
                        in_important_time = False
            counter += 1


        #Somehow the window following the transcript and the alignment are out of sync.
        if false_words_counter > 10:
            print("This should not happen.")
            print(transcript_file)

    def get_names(self):
        return self.names

    def get_important_times(self):
        return self.important_times

    def get_speaker_information_list(self):
        return self.speaker_information_list

    def get_sentence_boundaries(self):
        return self.sentence_boundaries

    def get_qanda(self):
        return self.qanda


'''
For every sound file, this cuts the important speeches into sentences and gets
MFCC and, for further information other features. It also saves the sentence ID
at which the Question-and-Answer section starts.
'''
def cut_to_mfcc():
    for file in os.listdir(directory):
        print(file)
        filename = os.fsdecode(file)[:-4]

        new_dir = FEATURES_DIRECTORY + filename + "/"

        if not os.path.isdir(new_dir):

            new_alignment = Sound_Data(filename)
            print(SOUND_DIRECTORY + filename + ".mp3")

            os.mkdir(new_dir)

            dir_wav = new_dir + "wav/"
            dir_arff = new_dir + "arff/"

            os.mkdir(dir_wav)
            os.mkdir(dir_arff)

            # print(filename)
            # print(new_alignment.get_names())
            print(new_alignment.get_important_times())

            speaker_information_list = new_alignment.get_speaker_information_list()
            print(speaker_information_list)

            splits = np.array(new_alignment.get_important_times())
            splits = np.multiply(splits, 1000)

            sentence_boundaries = np.array(new_alignment.get_sentence_boundaries())
            sentence_boundaries = np.multiply(sentence_boundaries, 1000)

            sound_file = AudioSegment.from_mp3(SOUND_DIRECTORY + filename + ".mp3")

            counter2 = 0

            qAndA = new_alignment.get_qanda() * 1000
            qAndASet = False


            for start, end in sentence_boundaries:
                x = start
                y = end

                # go through list of important times and check if the word boundaries are inside this list.
                gender_counter = 0
                for important_start, important_end in splits:
                    to_extract = False
                    if x > important_start and y < important_end:
                        to_extract = True
                    elif x > important_start and x < important_end:
                        y = important_end
                        to_extract = True
                    elif y > important_start and y < important_end:
                        x = important_start
                        to_extract = True

                    if to_extract:
                        print(
                            "Extracting Features for " + dir_arff + str(counter2) + "_" + speaker_information_list[
                                gender_counter])
                        new_file = sound_file[x: y]
                        new_file.export(
                            dir_wav + str(counter2) + "_" + speaker_information_list[gender_counter] + ".wav",
                            format="wav")
                        os.system("SMILExtract -C pre_processing/emobase2010.conf -I "
                                  + dir_wav + str(counter2) + "_" + speaker_information_list[
                                      gender_counter] + ".wav" + " -O "
                                  + dir_arff + str(counter2) + "_" + speaker_information_list[
                                      gender_counter] + ".arff" + " -l 0")
                        counter2 += 1
                        if not qAndASet:
                            if y > qAndA:
                                qAndASet = True

                                file = open(new_dir + "qAndACount.txt", "w")
                                file.write(str(counter2))
                                file.close()

                    gender_counter += 1


'''
Save change of EPS, Revenue and year over year results.
'''
def save_stock_information():
    transcripts_directory = os.fsencode(TRANSCRIPTS_DIRECTORY)
    for file in os.listdir(transcripts_directory):
        print(file[:-4])
        with open(transcripts_directory + file, "r") as to_read:
            lines = to_read.readlines()

            line = ""
            i = 0
            while i < 8:
                if "EPS of" in lines[i]:
                    line = lines[i]
                i += 1

            print(line)
        to_read.close()
        split_line = line.split(" ")
        percentage_EPS = 0
        percentage_Rev = 0
        year_over_year = 0
        try:
            EPS_planned = split_line[2].replace("$", "")
            EPS_beat = split_line[5].replace("$", "")
            Rev_planned = split_line[8].replace("$", "")
            Rev_beat = split_line[13].replace("$", "")
            year_over_year = split_line[9].replace("(", "")
            print(EPS_planned + " " + EPS_beat + " " + Rev_planned + " " + Rev_beat + " " + year_over_year)

            percentage_EPS = float(EPS_beat) / float(EPS_planned)
            print(percentage_EPS)

            Rev_planned_number = float(Rev_planned.replace("B", "").replace("M", ""))


            # if the numbers are given in 'Million' or 'Billion'
            if "B" in Rev_planned:
                Rev_planned_number *= 1000000000
            elif "M" in Rev_planned:
                Rev_planned_number *= 1000000
            Rev_beat_number = float(Rev_beat.replace("B", "").replace("M", ""))
            if "B" in Rev_beat:
                Rev_beat_number *= 1000000000
            elif "M" in Rev_beat:
                Rev_beat_number *= 1000000

            percentage_Rev = Rev_beat_number/Rev_planned_number

            year_over_year = float(year_over_year.replace("%", ""))/100

        except:
            print("No stock info found.")


        filename = file[:-4].decode('utf-8')
        try:
            with open(FEATURES_DIRECTORY + filename + "/earnings_numbers.txt", "w") as file_to_write:
                file_to_write.write(str(percentage_EPS) + "  " + str(percentage_Rev) + " " + str(year_over_year))
            file_to_write.close()
        except:
            print("ops, file not found.")

'''
Removes the wavs of the single sentences that have been generated on the
way to extract the MFCC-features.
'''
def remove_wavs():
    for file in os.listdir(FEATURES_DIRECTORY):
        print(file)
        if os.path.isdir(FEATURES_DIRECTORY + file + "/wav/"):
            shutil.rmtree(FEATURES_DIRECTORY + file + "/wav/")