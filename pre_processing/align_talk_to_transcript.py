# -----------------------------------------------------------
# This calls the forced alignment script for all transcripts.
#
# -----------------------------------------------------------


import os
from file_names import TRANSCRIPTS_DIRECTORY, SOUND_DIRECTORY

def align():
     transcripts_directory = TRANSCRIPTS_DIRECTORY
     sounds_directory = SOUND_DIRECTORY
     directory = os.fsencode(transcripts_directory)

     os.system("python3 gentle-master/serve.py")


     for file in os.listdir(directory):
          filename = os.fsdecode(file)
          text_file = transcripts_directory + filename
          print(text_file)
          sound_file = sounds_directory + filename[:-3] + "mp3"
          print(sound_file)
          if not os.path.isfile("pre_processing/alignments/" + filename[:-4] + "_alignment.json"):
               print(filename[:-4])
               os.system("python3 pre_processing/gentle-master/align.py " + sound_file + " " + text_file + " --output pre_processing/alignments/" + filename[:-4] + "_alignment.json")