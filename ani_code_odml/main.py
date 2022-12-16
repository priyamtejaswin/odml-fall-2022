import subprocess
import time
import os
import Utility as util

proc = subprocess.Popen(["python", "record.py"])  # start recording
base_directory = "somesearchdirectory"
while True:
    wav_files = util.searchForWavFiles(base_directory)
    if len(wav_files) == 0:
        time.sleep(200)
        break
    else:
        for file in wav_files:
            result = util.checkForWakeWord(file)
            if result:
                util.beginProcessing()
            else:
                os.remove(os.join(base_directory, file))
proc.terminate()  # very important, will otherwise continuously record!!
