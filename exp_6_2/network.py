import torch
import torch.nn as nn
import torch.nn.functional as F

# Alternating layers of ReLU and linear
class SimpleLinear(nn.Module):

    def __init__(self, layers, domain):
        super(SimpleLinear, self).__init__()
        self.layers = nn.ModuleList([l for l in layers if type(l) != nn.ReLU])
        self.domain = domain

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return -x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 6)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.fc2(x)
        return torch.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, z):
        x = self.relu(self.fc1(z))
        x = self.fc2(x)
        return x