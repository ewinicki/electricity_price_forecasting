import pandas as pd
import os
import torch

from itertools import chain

import FormatData as fd

def train(nn, dataloader, criterion, optimizer):
    samples = 0
    losses = 0.0

    nn.train()
    for data in dataloader:
        x, y = fd.create_variables(data)
        loss = criterion(nn(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        samples += x.shape[0]
        losses += loss.item()

    return losses / samples

def validate(nn, dataloader, criterion):
    samples = 0
    losses = 0.0

    with torch.no_grad():
        nn.eval()
        for data in dataloader:
            x, y = fd.create_variables(data)
            loss = criterion(nn(x), y)
            samples += x.shape[0]
            losses += loss.item()

        return losses / samples

def train_nn(nn, num_epochs, train_loader, validation_loader=None):
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(nn.parameters())

    epoch = 0
    results = {}
    results["training"] = []
    results["validation"] = []

    for epoch in range(num_epochs):
        if validation_loader is not None:
            results["validation"].append(validate(nn, validation_loader, criterion))

        results["training"].append(train(nn, train_loader, criterion, optimizer))

        print("epoch {}: training: {}, validation: {}".format(epoch,
            results["training"][-1], results["validation"][-1]))

    return pd.DataFrame(results)

def test_nn(nn, test_loader):
    criterion = torch.nn.MSELoss(reduction='mean')
    nn.train(False)
    nn.eval()
    y_pred_lst = []

    for data in test_loader:
        x, y = fd.create_variables(data)
        y_pred_lst.append(chain.from_iterable(nn(x).tolist()))

    return list(chain.from_iterable(y_pred_lst))
