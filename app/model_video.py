import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch import Tensor
from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.models import vgg16, densenet161


##############################
#         Encoder
##############################
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        # vgg = vgg16(pretrained=True) 
        # self.feature_extractor = nn.Sequential(*list(vgg.children())[:-1])

        # self.final = nn.Sequential(
        #     nn.Linear(vgg.classifier[6].in_features, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
        # )
        # self.final = nn.Sequential(
        #     nn.Linear(25088, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
        # )

        model = densenet161(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])
        self.final = nn.Sequential(
            nn.Linear(108192, latent_dim), nn.BatchNorm1d(latent_dim, momentum=0.01)
        )
    

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.final(x)

    
##############################
#           LSTM
##############################


class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x
    
    
##############################
#         ConvLSTM
##############################


class ConvLSTM_Video(nn.Module):
    def __init__(
        self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024, bidirectional=True, attention=True
    ):
        super(ConvLSTM_Video, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )
        self.attention = attention
        self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x)
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x)
        if self.attention:
            attention_w = F.softmax(self.attention_layer(x).squeeze(-1), dim=-1)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        return self.output_layers(x)
