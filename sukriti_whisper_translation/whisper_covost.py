import whisper
import os
import time
import random
import sys
random.seed(0)


import warnings
warnings.filterwarnings("ignore")

model = whisper.load_model("tiny.en") 

transcribed = ""
count = 0.0

clips_dir = sys.argv[1]
file_names = []
for root, dirs, files in os.walk(clips_dir, topdown=True):
    for file in files:
        if ".mp3" in file:
            file_names.append(file)

#file_names = random.choices(file_names, k=5000)

for file in file_names:
        result = model.transcribe(clips_dir + file)
        transcribed += file + '\t' + result['text'] + '\n'
        count += 1
        if count%500==0:
            print(count)

print("Done. Output saved in whisper_transcribed.txt.")
with open("whisper_transcribed.txt", "w") as text_file:
    print(transcribed, file=text_file)
