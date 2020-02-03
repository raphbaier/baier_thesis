# -----------------------------------------------------------
# Get the speaker names along with all the sentences they
# spoke.
#
# -----------------------------------------------------------


import os
import pandas as pd
from file_names import TRANSCRIPTS_DIRECTORY, FEATURES_DIRECTORY

def get_speeches_for_speakers():
    #the possible titles of the list of names of company members
    IMPORTANT_NAMES = ["Company Participants", "Executives", "Company Representatives", "Corporate Participants"]
    #list of important names ends here.
    UNIMPORTANT_NAMES = ["Analysts", "Conference Call Participants", "Analyst"]
    #End searching for names at the last name, operator
    OPERATOR_NAMES = ["Operator"]

    number_of_speakers = 0
    number_of_talks = 0
    company_participants = []
    participants_speeches = []


    for file in os.listdir(FEATURES_DIRECTORY):
        print(file)
        number_of_talks += 1

        with open(TRANSCRIPTS_DIRECTORY + file + ".txt", "r") as current_file:
            started = False
            ended = False

            current_name_counter = -1

            for line in current_file.readlines():
                current_line = line.replace("\n", "")
                if not ended:
                    if started:
                        if current_line in UNIMPORTANT_NAMES or current_line in OPERATOR_NAMES:
                            started = False
                            ended = True
                        elif current_line != "":
                            number_of_speakers += 1
                            name = current_line.replace("–", "-").split("-")[0]
                            print(name)
                            current_name_counter += 1
                            if name not in company_participants:
                                company_participants.append(name)
                                participants_speeches.append([])

                            arff_files_for_speaker = participants_speeches[company_participants.index(name)]
                            #if arff_files == None:
                             #   arff_files = []

                            for arff_file in os.listdir(FEATURES_DIRECTORY + file + "/arff/"):
                                #print(arff_file)
                                arff_id = arff_file[:-5].split("_")[2]
                                if arff_id == str(current_name_counter):
                                    arff_files_for_speaker.append(FEATURES_DIRECTORY + file + "/arff/" + arff_file)
                            participants_speeches[company_participants.index(name)] = arff_files_for_speaker

                    elif current_line in IMPORTANT_NAMES:
                        started = True

    print("The company participants:")
    print(company_participants)

    print("average speakers per earnings call: " + str(number_of_speakers/number_of_talks))

    #print(participants_speeches)

    stock_arff_df = pd.DataFrame(company_participants, columns=['participants'])

    participants_speeches_str = []
    for speeches in participants_speeches:
        string_to_add = str(speeches).replace("[", "").replace("]", "").replace("\'", "")
        participants_speeches_str.append(string_to_add)

    speeches_df = pd.concat([stock_arff_df, pd.DataFrame(participants_speeches_str, columns=['speeches'])], axis=1)

    speeches_df.head()
    speeches_df.to_csv("speakersAndTheirSpeeches" ".csv", mode='w', index=False, header=True)