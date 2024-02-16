import torch
import torch.nn as nn
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


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
