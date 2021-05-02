"""
Helper script for extracting frames from the UCF-101 dataset
"""

import av
import glob
import os
import time
import tqdm
import datetime
import argparse

dataset_path = "C:\\Users\\User\\Desktop\\Term 7 Deep Learning\\Big Project\\app\\video\\" 

def extract_frames(video_path):
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()


def extraction(filename):
   
    video_path = glob.glob(os.path.join(dataset_path+filename))

    sequence_type = "unknown"
    sequence_name = filename
    sequence_path = filename+"-frames"

    os.makedirs(sequence_path, exist_ok=True)

    print("video_path")
    print(video_path)

    extract_frames(video_path)

    # Extract frames
    for j, frame in enumerate(
        tqdm.tqdm(
            extract_frames(video_path[-1]),
        )
    ):
        frame.save(os.path.join(sequence_path, f"{j}.jpg"))

