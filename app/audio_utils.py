from __future__ import print_function
from moviepy.editor import *
import numpy as np
import scipy.io.wavfile as wavf
import numpy as np
import librosa
import sklearn
import soundfile as sf
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(dir_path, "video")

def extract_audio(filename):
    video_path = os.path.join(dataset_path, filename)
    vid = VideoFileClip('{}'.format(video_path))
    # vid = VideoFileClip('{}'.format(filename))
    # filename_trim = filename.split('\')[-1].split('.')[0]
    filename_trim = filename.split(".")[0]
    print(filename_trim)
    aud = vid.audio
    print("inside extract_audio: ", aud)
    aud.write_audiofile('./video/UCF_audio_raw/{}.wav'.format(filename_trim))



def pad_audio(filename, T):
    filename_trim = filename.split('/')[-1].split('.')[0]
    in_wav = './video/UCF_audio_raw/{}.wav'.format(filename_trim)
    
    print(filename_trim)
    out_wav = './video/UCF_audio_trimmed/{}.wav'.format(filename_trim)
    T = float(T)
    fs, in_data = wavf.read(in_wav)
    # Calculate target number of samples
    N_tar = int(fs * T)
    # Calculate number of zero samples to append
    shape = in_data.shape
    # Create the target shape

    if shape[0] < N_tar:
        N_pad = N_tar - shape[0]
        print(filename_trim)
        print("Padding with %s seconds of silence" % str(N_pad/fs) )
        shape = (N_pad,) + shape[1:]


        # Stack only if there is something to append
        if shape[0] > 0:
            if len(shape) > 1:
                output = np.vstack((np.zeros(shape, in_data.dtype), in_data))
            else:
                output = np.hstack((np.zeros(shape, in_data.dtype), in_data))

        print(output.shape)

    else:
        output = in_data[0:N_tar, :]

    wavf.write(out_wav, fs, output)
    

def extract_features(filename):
    filename_trim = filename.split('/')[-1].split('.')[0]
    print("extract_feature filename: ", filename_trim)
    y, sr = librosa.load('./video/UCF_audio_trimmed/{}.wav'.format(filename_trim), mono=True, duration=20)

    rmse = librosa.feature.rms(y=y) # (1, 87) array
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr) # (12, 87) array
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr) #(1,87) array
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr) # (1,87)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr) # (1, 87)
    zcr = librosa.feature.zero_crossing_rate(y) #(1, 87)
    mfcc = librosa.feature.mfcc(y=y, sr=sr) # (20, 87)

    features_vec = np.concatenate((rmse[0], chroma_stft.mean(axis=0),
                                   spec_cent[0], spec_bw[0], rolloff[0],
                                   zcr[0], mfcc.mean(axis=0)))

    return features_vec

