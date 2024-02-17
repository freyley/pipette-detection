import torch
import torch.nn as nn
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Output: 16x500x500
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # Output: 16x250x250
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 250 * 250 + 500 * 500, 120)
        self.fc2 = nn.Linear(120, 3)  # Output layer for triplet

    def forward(self, x):
        residual = self.flatten(x)

        # Convolution + ReLU + Pooling
        out = self.pool(F.relu(self.conv1(x)))
        # Flatten for fully connected layer
        out = self.flatten(out)

        # Combine residual
        out = torch.cat((out, residual), dim=1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


def get_detector():
    return Detector()

def get_effnet_detector():
    # Load the pretrained model
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)

    # Adjust the first convolutional layer for 1-channel grayscale input
    first_conv_layer = model.features[0][0]
    model.features[0][0] = nn.Conv2d(1, first_conv_layer.out_channels,
                                     kernel_size=first_conv_layer.kernel_size, stride=first_conv_layer.stride,
                                     padding=first_conv_layer.padding, bias=False)

    # Change the output layer to produce 3 float outputs
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    return model

def load_model_weights(model, filename):
    if os.path.isfile(filename):
        model.load_state_dict(torch.load(filename))
        print("Loaded model weights from:", filename)
    else:
        print("No weights file found.")
    return model

def save_model_weights(model, filename):
    torch.save(model.state_dict(), filename)

