# import queue
from collections import deque
import numpy as np
import torch
import soundfile as sf
import sys
from generate_ts import wrapper as asr_model
import os
import csv
import sys
from tqdm import tqdm
import pandas as pd

from transformers import pipeline
en2de_model = pipeline("translation_en_to_de", "Helsinki-NLP/opus-mt-en-de")

# GLOBALS, because I'm lazy.
state = None
hypo = None
previous_right_context = None
data_queue = deque()

def transcribe(np_array, should_print=True):
    global state, hypo
    tensor = torch.tensor(np_array)
    # print(tensor.shape)
    transcript, hypo, state = asr_model(tensor, hypo, state)
    if should_print and transcript:
        print(transcript, end="", flush=True)
        return transcript
    else:
        return ""

def process(should_print=True):
    global previous_right_context, data_queue
    if previous_right_context is None:
        try:
            previous_right_context = [
                data_queue.popleft() for _ in range(1)
            ]
        except IndexError as error:
            data_queue = deque()
            return ''

    # Get 4 segments.
    try:
        segments = [
            data_queue.popleft() for _ in range(4)
        ]
    except IndexError as error:
        data_queue = deque()
        return ''

    current_input = previous_right_context + segments

    with torch.no_grad():
        text = transcribe(np.concatenate(current_input), should_print=should_print)

    # Save right context.
    previous_right_context = current_input[-1:]
    return text

def asr_on_file(path):
    global state, hypo, previous_right_context, data_queue
    state, hypo, previous_right_context = None, None, None
    data_queue = deque()

    count = 0
    for data in sf.blocks(path, blocksize=640, dtype='float32'):
        # data = block.tobytes()
        data_queue.append(data)
        count += 1

    while count % 5 != 0:
        data_queue.append(data)
        count += 1

    # Start timing here.
    asr = []
    while data_queue:
        asr.append(process())
    # End timing here.

    return ' '.join(asr)

print("Initializing model with sample audio ...")
sample_wav = "./download.wav"

for _ in range(2):
    asr_on_file(sample_wav)
    print('...')

ENDE_ROOT = "./covost2_en_de"

print("Starting CoVoST2 evaluation.")
with open(sys.argv[1]) as fp:
    # reader = csv.DictReader(fp, delimiter='\t')
    reader = df = pd.read_csv(fp, sep="\t", header=0, encoding="utf-8", escapechar="\\", quoting=csv.QUOTE_NONE, na_filter=False)
    clips = [os.path.join(ENDE_ROOT, "clips_dev", "waves", r) for r in reader["path"]]
    clips = [os.path.splitext(p)[0] + '.wav' for p in clips]

print("Found %d clips." % len(clips))

eval_eng = []
eval_ger = []

for path in tqdm(clips):
    text_eng = asr_on_file(path)
    text_ger = en2de_model(text_eng)[0]["translation_text"]

with open(os.path.join(ENDE_ROOT, "stream_helsinki_en.txt"), 'w') as fp:
    writer = csv.writer(fp)
    writer.writerows(eval_eng)

print("Written eng.")

with open(os.path.join(ENDE_ROOT, "stream_helsinki_de.txt"), 'w') as fp:
    writer = csv.writer(fp)
    writer.writerows(eval_ger)

print("Written ger.")

print("Check outputs in %s" % ENDE_ROOT)