import librosa
import pandas as pd
import numpy as np
from audio_utils import *
from model_video import *
from model_audio import *
from extract_frames import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import operator
import os
import pathlib
import csv
from audio_utils import *

import warnings
warnings.filterwarnings('ignore')

VIDEO_WEIGHT_PATH = "./model_weights/ConvLSTM_14_dense.pth"
AUDIO_WEIGHT_PATH = "./model_weights/ConvLSTM_14_audio.pth"

dir_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(dir_path, "video")


def infer_video(filename):
    filename_short = filename.split('.')[0]

    image_shape = (3, 224, 224)
    split_path = "train_test_split"
    split_number = 1
    sequence_length = 50
    batch_size = 4
    latent_dim = 256
    num_epochs = 15
    checkpoint_model = ""
    checkpoint_interval = 1
    num_classes = 49

    model = ConvLSTM_Video(
        num_classes=num_classes,
        latent_dim=latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True)
    model.load_state_dict(torch.load(
        VIDEO_WEIGHT_PATH, map_location=torch.device('cpu')))
    model = model
    model.eval()

    labels = {'ApplyEyeMakeup': 1, 'ApplyLipstick': 2, 'Archery': 3, 'BabyCrawling': 4, 'BalanceBeam': 5, 'BandMarching': 6, 'BlowDryHair': 7, 'BlowingCandles': 8, 'BodyWeightSquats': 9, 'Bowling': 10, 'BoxingPunchingBag': 11, 'BoxingSpeedBag': 12, 'BrushingTeeth': 13, 'CliffDiving': 14, 'CricketBowling': 15, 'CricketShot': 16, 'CuttingInKitchen': 17, 'FieldHockeyPenalty': 18, 'FloorGymnastics': 19, 'FrisbeeCatch': 20, 'FrontCrawl': 21, 'Haircut': 22, 'Hammering': 23, 'HammerThrow': 24,
              'HandstandWalking': 25, 'HeadMassage': 26, 'IceDancing': 27, 'Knitting': 28, 'LongJump': 29, 'MoppingFloor': 30, 'ParallelBars': 31, 'PlayingCello': 32, 'PlayingDaf': 33, 'PlayingDhol': 34, 'PlayingFlute': 35, 'PlayingSitar': 36, 'Rafting': 37, 'ShavingBeard': 38, 'Shotput': 39, 'SkyDiving': 40, 'SoccerPenalty': 41, 'StillRings': 42, 'SumoWrestling': 43, 'Surfing': 44, 'TableTennisShot': 45, 'Typing': 46, 'UnevenBars': 47, 'WallPushups': 48, 'WritingOnBoard': 49}

    input_shape = (3, 244, 224)

    transform = transforms.Compose(
        [
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    video_path = os.path.join(dataset_path, filename)

    # Extract frames as tensors
    image_sequence = []
    count_frames = 0
    for frame in tqdm.tqdm(extract_frames(video_path), desc="Processing frames"):
        count_frames += 1
        if count_frames > 40:
            break
        image_tensor = Variable(transform(frame))
        image_sequence.append(image_tensor)

    image_sequence = torch.stack(image_sequence)
    image_sequence = image_sequence.view(1, *image_sequence.shape)
    print(image_sequence.shape)
    # Get label prediction for frame
    with torch.no_grad():
        prediction = model(image_sequence)
        # print(prediction)
        val = prediction.argmax(1).item()
        prob = torch.max(prediction).item()
        predicted_label = list(labels.keys())[
            list(labels.values()).index(val+1)]
        print(predicted_label)
        print(prob)

    return predicted_label, prob




def infer_audio(filename):
    video_path = os.path.join(dataset_path, filename)
    print("videoPath: ", video_path)

    # extacting video features
    extract_audio(filename)
    pad_audio(filename, 2)

    features_vec = extract_features(filename)
    feature_tensor = torch.Tensor(features_vec)
    feature_tensor = feature_tensor.view(1, *feature_tensor.shape)

    print("feature tensor: ", feature_tensor.shape)

    
    # Extract image frames as tensors
    labels = {'ApplyEyeMakeup': 1, 'ApplyLipstick': 2, 'Archery': 3, 'BabyCrawling': 4, 'BalanceBeam': 5, 'BandMarching': 6, 'BlowDryHair': 7, 'BlowingCandles': 8, 'BodyWeightSquats': 9, 'Bowling': 10, 'BoxingPunchingBag': 11, 'BoxingSpeedBag': 12, 'BrushingTeeth': 13, 'CliffDiving': 14, 'CricketBowling': 15, 'CricketShot': 16, 'CuttingInKitchen': 17, 'FieldHockeyPenalty': 18, 'FloorGymnastics': 19, 'FrisbeeCatch': 20, 'FrontCrawl': 21, 'Haircut': 22, 'Hammering': 23, 'HammerThrow': 24,
              'HandstandWalking': 25, 'HeadMassage': 26, 'IceDancing': 27, 'Knitting': 28, 'LongJump': 29, 'MoppingFloor': 30, 'ParallelBars': 31, 'PlayingCello': 32, 'PlayingDaf': 33, 'PlayingDhol': 34, 'PlayingFlute': 35, 'PlayingSitar': 36, 'Rafting': 37, 'ShavingBeard': 38, 'Shotput': 39, 'SkyDiving': 40, 'SoccerPenalty': 41, 'StillRings': 42, 'SumoWrestling': 43, 'Surfing': 44, 'TableTennisShot': 45, 'Typing': 46, 'UnevenBars': 47, 'WallPushups': 48, 'WritingOnBoard': 49}

    input_shape = (3, 244, 224)
    transform = transforms.Compose(
        [
            transforms.Resize(input_shape[-2:], Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_sequence = []
    count_frames = 0
    for frame in tqdm.tqdm(extract_frames(video_path), desc="Processing frames"):
        count_frames += 1
        if count_frames > 40:
            break
        image_tensor = Variable(transform(frame))
        image_sequence.append(image_tensor)

    image_sequence = torch.stack(image_sequence)
    image_sequence = image_sequence.view(1, *image_sequence.shape)


    image_shape = (3, 224, 224)
    split_path = "train_test_split"
    split_number = 1
    sequence_length = 50
    batch_size = 4
    latent_dim = 256
    num_epochs = 15
    checkpoint_model = ""
    checkpoint_interval = 1
    num_classes = 49

    model = ConvLSTM_Audio(
        num_classes=num_classes,
        latent_dim=latent_dim,
        lstm_layers=1,
        hidden_dim=1024,
        bidirectional=True,
        attention=True)
    model.load_state_dict(torch.load(
        AUDIO_WEIGHT_PATH, map_location=torch.device('cpu')))
    model = model
    model.eval()



    # passing to the model with both image tensors + audio tensors
    with torch.no_grad():
        prediction = model(image_sequence, feature_tensor)
        # print(prediction)
        val = prediction.argmax(1).item()
        prob = torch.max(prediction).item()
        predicted_label = list(labels.keys())[
            list(labels.values()).index(val+1)]
        print(predicted_label)
        print(prob)

    return predicted_label, prob



