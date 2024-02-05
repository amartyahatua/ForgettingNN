import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MyLayer(nn.Module):
    """
    This is a user defined neural network layer
    param: int input size
    param: tensor Input * random weight
    """
    def __init__(self, in_size):
        super().__init__()
        self.weight = None
        self.size = in_size
        self.epoch = None
        self.rank = None

    def forward(self, x, n_epochs, rank):
        self.rank = rank
        self.epoch = n_epochs
        self.weight = np.exp(-self.epoch / self.r)
        return x * self.weight


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.activation_1 = nn.ReLU()
        self.activation_2 = nn.ReLU()
        self.activation_3 = nn.ReLU()
        self.activation_4 = nn.ReLU()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.epoch = 0
        self.rank = 0

    def forward(self, x):
        x= self.conv1(x)
        x = self.activation_1 (x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.activation_2(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc1(x)
        x = self.activation_3(x)
        x = MyLayer(x, self.epoch, self.rank)

        x = self.fc2(x)
        x = self.activation_4(x)
        x = MyLayer(x, self.epoch, self.rank)
        x = self.fc3(x)
        return x


