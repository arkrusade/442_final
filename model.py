import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Displays a progress bar

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split



N = 43  # 43 classes in GTSRB


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # in_channels, out_channesl, kernel_size, stride, padding
        xy1 = 3

        i1 = 1

        o1 = 32
        k1 = 5
        s1 = 1
        p1 = 1

        # o = floor( ( i + 2p - k ) / s )  + 1
        xy2 = dims(xy1, p1, k1, s1)

        i2 = o1
        o2 = 64
        k2 = 3
        s2 = 1
        p2 = 1

        xy3 = dims(xy2, p2, k2, s2)

        self.conv1 = nn.Conv2d(i1, o1, k1, s1, p1)

        self.conv2 = nn.Conv2d(i2, o2, k2, s2, p2)

        k3 = 3
        s3 = k3
        p3 = 1
        xy4 = dims(xy3, p3, k3, s3)
        self.pool = nn.MaxPool2d(kernel_size=k3, padding=p3)

        self.flattened = o2 * xy4 * xy4
        self.fc1 = nn.Linear(self.flattened, 32)
        self.fc2 = nn.Linear(32, N)
        self.dropfc = nn.Dropout(p=0.5)

    def forward(self, x):
        relu = nn.ReLU()
        x = self.conv1(x)
        x = relu(x)
        x = self.conv2(x)
        x = relu(x)
        x = self.pool(x)
        x = x.view(-1, self.flattened)  # Flatten each image in the batch
        x = self.fc1(x)
        x = relu(x)
        x = self.dropfc(x)
        x = self.fc2(x)
        x = nn.Softmax(x)
        return x

