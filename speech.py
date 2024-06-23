import json
import os
import pyperclip
import time
import signal
from inference import *
import subprocess
def get_key():
    with open("vars.json", "r") as file:
        data = json.load(file)
        return data["api_key"]

def send_interrupt(process):
    process.kill()

def voice_input(interval):
    command = "whisper-stream -t " + get_key() + " -d " + str(interval)
    process = subprocess.Popen(command, shell=True)

    # process = subprocess.Popen(['whisper-stream', '-t', get_key(), '-d', str(interval)])
    time.sleep(interval)
    send_interrupt(process)
    transcription = pyperclip.paste()
    print(transcription)
    return transcription

# def loop(interval):
#     threading.Timer(3.0, search_keyword(voice_input(interval), used_labels)).start()

def search_keyword(text, labels_lst):
    for label in labels_lst:
        if text.find(label) != -1:
            return label
    return None

def loop(interval):
    while True:
        words = voice_input(interval)
        # label = search_keyword(words)
        # if label is not None:
        #     return label
        print("Not Yet")
        break
        
print(loop(3))