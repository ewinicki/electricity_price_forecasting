#!/usr/bin/python3

import os
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt

import torch
from CheckCuda import cuda_available

from TrainTest import train_nn
from TrainTest import test_nn

import FormatData as fd
from Model import Model
from CNN import CNN
from FFNN import FFNN
from DateRange import DateRange, DateRanges
from Results import TrainResults, TestResults

# PREDICTED VALUE
TEST_VALUE = 'RT_LMP'

# lag days to add to daily data
LAG = [1, 2, 7, 30, 365]

OUTPUTS = 24

# data conditioning variables
BATCH_SIZE = 4

def main(data, epochs, optimize, model, span, variance, write):
    # import index slicing for pandas dataframes
    dates = DateRanges(span)
    model = Model(model)

    # import data from files
    ts = pd.read_pickle(data)
    ts = ts.loc[dates.range.hours]
    ts.index.name = 'Date'

    # pull expected results from data
    y = ts[[TEST_VALUE]]
    y = ts[TEST_VALUE]

    test_results = TestResults()
    test_results.y = y.loc[dates.test.hours]

    if model.category == "nn":
        # adjust training start to match lag days
        dates.train.start = dates.train.start + pd.to_timedelta(max(LAG), unit='d')
        train_results = TrainResults(epochs)

        # generate and store NNs for supervised learning data set
        if model.classification == "ffnn":
            # generate supervised learning time series data
            sl_data = fd.ffnn_data(ts, LAG)

            # new start date after lag
            actual = fd.to_day(y).loc[dates.range.hours]
            sl_data = sl_data.drop('CURRENT', axis=1, level=0).unstack()
            dl_train, dl_validation, dl_test, inputs = fd.create_dls(sl_data,
                    actual, variance, BATCH_SIZE, TEST_VALUE, dates)

            if optimize > 1:
                for sample in range(optimize):
                    print("training sample: " + str(sample))
                    nn = FFNN(inputs, **model.parameters, outputs=OUTPUTS)
                    nn = cuda_available(nn)
                    train_results.add(train_nn(nn, epochs, dl_train, dl_validation))

            nn = FFNN(inputs, **model.parameters, outputs=OUTPUTS)


        # generate and store CNNs for channel style data
        elif model.classification == "cnn":
            # generate with data points as channel values
            cnn_data = fd.cnn_data(ts, LAG)
            actual = fd.to_day(y).loc[dates.range.hours]
            dl_train, dl_validation, dl_test, seq_len = fd.create_dls(cnn_data, actual,
                    variance, BATCH_SIZE, TEST_VALUE, dates)

            # generate neural networks
            channels = len(cnn_data.index.levels[-1])
            # seq_len = cnn_data.shape[1]

            if optimize > 1:
                for sample in range(optimize):
                    print("training sample: " + str(sample))
                    nn = CNN(channels, seq_len, **model.parameters)
                    nn = cuda_available(nn)
                    train_results.add(train_nn(nn, epochs, dl_train, dl_validation))

            nn = CNN(channels, seq_len, **model.parameters)

        nn = cuda_available(nn)
        print("training test network")
        if optimize > 1:
            train_nn(nn, train_results.optimal, dl_train, dl_validation)
        else:
            train_results.add(train_nn(nn, epochs, dl_train, dl_validation))

        test_results.yhat = test_nn(nn, dl_test)

    if write is not None:
        train_results.plot(write + "_train.png", show=False)
        if optimize > 1:
            train_results.kde(write + "_kde.png", show=False)
        train_results.write_optimal(write + "_epochs")
        train_results.to_msgpack(write + "_train.msg")
        model.to_excel(write + ".xlsx")
        test_results.plot(write + "_test.png", show=False)
        test_results.to_excel(write + ".xlsx")
        test_results.to_msgpack(write + "_test.msg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data file')

    parser.add_argument('-e', '--epochs', help='number of epochs')
    parser.add_argument('-o', '--optimize', default=1, help='number of samples')
    parser.add_argument('-m', '--model', help='model file')
    parser.add_argument('-s', '--span', help='date range')
    parser.add_argument('-v', '--variance', help='pca variance')
    parser.add_argument('-w', '--write', help='write results')

    args = parser.parse_args()

    main(args.data, int(args.epochs), int(args.optimize), args.model,
            args.span, float(args.variance), args.write)
