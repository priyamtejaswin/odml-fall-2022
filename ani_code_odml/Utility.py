import os


def searchForWavFiles(search_directory):
    wav_files = []
    for file in os.listdir(search_directory):
        if file.endswith("wav"):
            wav_files.append(os.path.join(search_directory, file))
    return wav_files


def checkForWakeWord(wav_file):
    return False


def beginProcessing():
    return

