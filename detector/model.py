import os
import ssl

import certifi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    efficientnet_v2_m, vit_b_16,
    efficientnet_v2_l, vit_l_16, vit_l_32,
    maxvit_t,
)


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


ssl._create_default_https_context = ssl._create_unverified_context
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


def get_detector():
    return Detector()


def get_effnet_detector(no_pretrained=False, size=None):
    # Load the pretrained model
    if size is None or size in ('medium', 'm'):
        modelClass = efficientnet_v2_m
    elif size in ('large', 'l'):
        modelClass = efficientnet_v2_l
    else:
        raise ValueError("Invalid size")

    if no_pretrained:
        model = modelClass()
    else:
        model = modelClass(weights="DEFAULT")

    # Adjust the first convolutional layer for 1-channel grayscale input
    # first_conv_layer = model.features[0][0]
    # model.features[0][0] = nn.Conv2d(1, first_conv_layer.out_channels,
    #                                  kernel_size=first_conv_layer.kernel_size, stride=first_conv_layer.stride,
    #                                  padding=first_conv_layer.padding, bias=False)
    #
    # Change the output layer to produce 3 float outputs
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model.get_last_layer = lambda: model.classifier[1]
    return model


def get_vt_detector(no_pretrained=False, size=None):
    if size is None or size in ('b16', 'b_16'):
        modelClass = vit_b_16
    elif size in ('l16', 'l_16'):
        modelClass = vit_l_16
    elif size in ('l32', 'l_32'):
        modelClass = vit_l_32
    else:
        raise ValueError("Invalid size")

    if no_pretrained:
        model = modelClass()
    else:
        model = modelClass(weights='DEFAULT')

    final_layer = list(model.heads.children())[-1]

    num_features = final_layer.in_features
    model.heads[-1] = nn.Linear(num_features, 3)
    model.get_last_layer = lambda: model.heads[-1]
    return model

def get_maxvit_detector(no_pretrained=False):
    if no_pretrained:
        model = maxvit_t()
    else:
        model = maxvit_t(weights='DEFAULT')

    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 3)
    model.get_last_layer = lambda: model.classifier[-1]

    return model

def save_checkpoint(model, optimizer, scheduler, epoch, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, scheduler, filename):
    if not os.path.isfile(filename):
        print("No weights file found.")
        return 0
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    return epoch



def load_model_weights(model, filename):
    if os.path.isfile(filename):
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(filename, map_location=map_location))
        print("Loaded model weights from:", filename)
    else:
        print("No weights file found.")
    return model


def save_model_weights(model, filename):
    torch.save(model.state_dict(), filename)
