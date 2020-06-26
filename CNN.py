import torch
from torch.autograd import Variable
import torch.nn as nn

from math import floor

class CNN(nn.Module):
    def __init__(self, channels, seq_len, f1, f2, h1, h2, k1, k2,
            p1, p2, g1, g2, s1, s2, outputs=24):
        super(CNN, self).__init__()
        self.h2 = h2

        # create convolutional and pool layers
        self.conv1 = nn.Conv1d(channels, f1, k1, stride=s1, groups=g1)
        self.pool1 = nn.MaxPool1d(p1)

        self.conv2 = nn.Conv1d(f1, f2, k2, stride=s2, groups=g2)
        self.pool2 = nn.MaxPool1d(p2)

        # output from conv1 and pool1
        c1_out = self.new_seq(seq_len, k1, s1)
        p1_out = self.new_seq(c1_out, p1, p1)

        # output from conv2 and pool2
        c2_out = self.new_seq(p1_out, k2, s2)
        p2_out = self.new_seq(c2_out, p2, p2)

        # create fully connected output layer
        self.fc1 = nn.Linear(f2 * p2_out, h1)

        # single hidden layer
        if h2 is None:
            self.fc2 = nn.Linear(h1, outputs)
        # two hidden layers
        else:
            self.fc2 = nn.Linear(h1, h2)
            self.fc3 = nn.Linear(h2, outputs)


    def forward(self, x):
        # two sequenced convolutional layers
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))

        # convert 3D tensor back to 2D tensor
        x = x.view(x.shape[0], -1)

        # fully connected feed forward neural network
        y_pred = self.fc2(torch.tanh(self.fc1(x)))

        if self.h2 is not None:
            y_pred = self.fc3(torch.tanh(y_pred))

        return y_pred

    def new_seq(self, seq_len, kernel, stride=1):
        return floor((seq_len - kernel) / stride + 1)
