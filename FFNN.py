import torch
import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, inputs, h1, h2, outputs=24):
        super(FFNN, self).__init__()
        self.h2 = h2

        self.fc1 = nn.Linear(inputs, h1)

        if self.h2 is None:
            self.fc2 = nn.Linear(h1, outputs)
        else:
            self.fc2 = nn.Linear(h1, h2)
            self.fc3 = nn.Linear(h2, outputs)

    def forward(self, x):
        if self.h2 is None:
            return self.fc2(torch.tanh(self.fc1(x)))
        else:
            return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))
