# -----------------------------------------------------------
# Save earnings calls transcripts by accessing the GUI of
# google.com or seekingalpha.com
#
# -----------------------------------------------------------

import csv
import os
import time

import cv2
import numpy as np
import pyautogui
from pynput import keyboard

from file_names import TRANSCRIPTS_DIRECTORY

transcripts_directory = "../" + TRANSCRIPTS_DIRECTORY


# From https://github.com/drov0/python-imagesearch/blob/master/imagesearch.py
def _imagesearch(image, precision=0.8):
    im = pyautogui.screenshot()
    # im.save('testarea.png') usefull for debugging purposes, this will save the captured region as "testarea.png"
    img_rgb = np.array(im)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(image, 0)
    template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val < precision:
        return [-1, -1]
    return max_loc


def _on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))


def _on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False


'''
perform the search inside the https://seekingalpha.com - site
For this script to work, we set the resolution to 1920x1080px
and used internet explorer maximally expanded. If it doesn't
work as planned for you, you might want to adjust the base
screenshots and pixel values this script is working with.
'''
def perform_search_in_alpha_gui():
    time_delay = 1

    with open('ma_stocks.csv', mode='r') as infile:  # contains company's ticker symbols
        reader = csv.reader(infile)
        with open('ma_stocks_new.csv', mode='w') as outfile:
            writer = csv.writer(outfile)
            abb_to_stock = {rows[0]: rows[1] for rows in reader}

    print(abb_to_stock['A'])

    counter = 0

    # when to start and to stop
    start_count = 0
    max_count = 30000

    time.sleep(3 * time_delay)
    with open("filename1.txt") as f:
        content = f.readlines()
        for line in content:
            if start_count < counter < max_count:
                earnings_info = line
                earnings_infos = earnings_info[:-5].split("-")
                print(earnings_infos)
                if not os.path.isfile('transcripts/' + earnings_info[:-5] + '.txt'):

                    pos = _imagesearch.imagesearch("search_box.png")
                    if pos[0] != -1:
                        pyautogui.moveTo(pos[0], pos[1])

                    pyautogui.click()

                    pyautogui.typewrite(abb_to_stock[earnings_infos[0]] + " (")

                    pyautogui.typewrite(earnings_infos[0] + ")")

                    pyautogui.typewrite(" ")
                    pyautogui.typewrite(earnings_infos[2])
                    pyautogui.typewrite(" ")
                    pyautogui.typewrite(earnings_infos[1])
                    pyautogui.typewrite(" - earnings call transcript")

                    time.sleep(0.2 * time_delay)
                    pyautogui.press('enter')

                    time.sleep(2.8 * time_delay)
                    pyautogui.moveTo(709, 369)
                    pyautogui.click()

                    time.sleep(2)
                    pos = _imagesearch.imagesearch("search_for_transcript.png")
                    if pos[0] != -1:
                        print("position : ", pos[0], pos[1])
                        pyautogui.moveTo(pos[0], pos[1])
                        pyautogui.click()

                    time.sleep(2 * time_delay)
                    pyautogui.moveTo(600, 246)
                    pyautogui.mouseDown()
                    time.sleep(0.5 * time_delay)
                    pyautogui.moveTo(736, 379)
                    pyautogui.scroll(-3000)
                    pyautogui.mouseUp()

                    transcript_to_save = os.popen('xsel').read()
                    print(transcript_to_save)
                    with open('transcripts/' + earnings_info[:-5] + '.txt', 'w') as file_to_write:
                        file_to_write.write(transcript_to_save)
                    file_to_write.close()

                    counter += 1
                    time.sleep(1 * time_delay)

            counter += 1
    print(counter)

'''
Perform a search in the google search bar
'''
def perform_search_in_google():
    time_delay = 1

    with open('ma_stocks.csv', mode='r') as infile:
        reader = csv.reader(infile)
        with open('ma_stocks_new.csv', mode='w') as outfile:
            abb_to_stock = {rows[0]: rows[1] for rows in reader}

    print(abb_to_stock['A'])

    counter = 0

    # when to start and to stop
    start_count = 0
    max_count = 2300

    time.sleep(3 * time_delay)
    with open("filename1.txt") as f:
        content = f.readlines()
        for line in content:
            if start_count < counter < max_count:
                earnings_info = line
                earnings_infos = earnings_info[:-5].split("-")
                print(earnings_infos)
                if not os.path.isfile('transcripts/' + earnings_info[:-5] + '.txt'):
                    pyautogui.click(977, 444)

                    pyautogui.typewrite("site:seekingalpha.com ")
                    pyautogui.typewrite(earnings_infos[0])

                    pyautogui.typewrite(" ")
                    pyautogui.typewrite(earnings_infos[2])
                    pyautogui.typewrite(" ")
                    pyautogui.typewrite(earnings_infos[1])
                    pyautogui.typewrite(" results earnings call AND intitle:transcript")

                    time.sleep(0.2 * time_delay)
                    pyautogui.press('enter')

                    with keyboard.Listener(
                            on_press=_on_press,
                            on_release=_on_release) as listener:
                        listener.join()

                    pyautogui.moveTo(600, 246)
                    pyautogui.mouseDown()
                    time.sleep(0.5 * time_delay)
                    pyautogui.moveTo(736, 379)
                    pyautogui.scroll(-3000)
                    pyautogui.mouseUp()

                    transcript_to_save = os.popen('xsel').read()
                    print(transcript_to_save)
                    with open('transcripts/' + earnings_info[:-5] + '.txt', 'w') as file_to_write:
                        file_to_write.write(transcript_to_save)
                    file_to_write.close()

                    pyautogui.click(845, 81)
                    time.sleep(0.1)
                    pyautogui.typewrite("www.google.de")
                    pyautogui.press('enter')
                    time.sleep(0.5)

                    time.sleep(1 * time_delay)
            counter += 1
    print(counter)

'''
Check if there can be found some wrong transcripts and remove them.
'''
def _remove_wrong_transcripts():
    for filename in os.listdir(transcripts_directory):
        file_location = transcripts_directory + "/" + filename
        with open(file_location) as transcript_to_check:
            transcript_data = transcript_to_check.read()
            transcript_title = ""
            if os.stat(file_location).st_size != 0 and os.stat(file_location).st_size != 1:
                transcript_title = transcript_data.split("\n")[0]

            keep_string = True
            if ("(" + filename.split("-")[0] + ") ") not in transcript_title:
                keep_string = False
            if (filename.split("-")[2][:-4] + " " + filename.split("-")[
                1] + " Results - Earnings Call Transcript") not in transcript_title:
                keep_string = False
            if len(transcript_data) < 10000:
                keep_string = False

            if "Accenture" in transcript_title or "EarningsCallTranscript" in transcript_title:
                keep_string = True
            if "F1Q" in transcript_title:
                keep_string = True
            if "F2Q" in transcript_title:
                keep_string = True
            if "F3Q" in transcript_title:
                keep_string = True
            if "F4Q" in transcript_title:
                keep_string = True
            if "AMVMF" in transcript_title:
                keep_string = True

            if not keep_string:
                # os.remove(file_location)
                print("Removed the file " + filename)
                print(filename)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--gui', required=True,
                        help='argument for the type of the GUI used (seekingalpha or google)')
    args = parser.parse_args()
    if args.gui == 'seekingalpha':
        perform_search_in_alpha_gui()
        _remove_wrong_transcripts()
    elif args.gui == 'google':
        perform_search_in_google()
        _remove_wrong_transcripts()
    else:
        print("Please define the GUI argument as \'seekingalpha\' or \'google\'.")
