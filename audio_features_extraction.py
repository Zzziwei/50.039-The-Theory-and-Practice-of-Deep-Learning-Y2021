from __future__ import print_function
from moviepy.editor import *
import numpy as np
import scipy.io.wavfile as wavf

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import sklearn
import soundfile as sf
import IPython.display as ipd
import os
from audio_utils import *
import pickle

with open('dirs_with_audio.pickle', 'rb') as handle:
    have_audio_dirs = pickle.load(handle)

# Separate the audio files from the video
for i in have_audio_dirs:
    sub_dir = ("./UCF_49/{}".format(i))
    file_list = os.listdir(sub_dir)
    for f in file_list:
        file_path = "{}/{}".format(sub_dir, f)
        extract_audio(file_path)

raw_audio_dir = "./UCF_audio_raw"

for i in os.listdir(raw_audio_dir):
    file_path = ("./UCF_audio_raw/{}".format(i))
    pad_audio(file_path, 2)

audio_features_dict = {}

trim_audio_dir = "./UCF_audio_trimmed"

for audio_file in os.listdir(trim_audio_dir):
    print(audio_file)
    file_path = ("./UCF_audio_trimmed/{}".format(audio_file))
    features = extract_features(file_path)
    audio_features_dict[audio_file] = features

with open('audio_features.pickle', 'wb') as handle:
    pickle.dump(audio_features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)