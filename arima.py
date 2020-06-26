#!/usr/bin/python3

import os
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_model import ARIMA
from sklearn.decomposition import PCA

from Model import Model

import FormatData as fd
from Arima import Arima
from DateRange import DateRange, DateRanges
from Results import TestResults

# PREDICTED VALUE
TEST_VALUE = 'RT_LMP'

# lag days to add to daily data
DAY_HRS = 24
YEAR_HRS = 365 * 24

OUTPUTS = 24

def main(data, model, span, variance, write):
    # import index slicing for pandas dataframes
    dates = DateRanges(span)
    model = Model(model)

    # import data from files
    ts = pd.read_pickle(data)
    ts = ts.loc[dates.range.hours]
    ts.index.name = 'Date'

    # pull expected results from data
    y = ts[TEST_VALUE]

    test_results = TestResults()
    test_results.y = y.loc[dates.test.hours]

    # create arima object
    arima = Arima(**model.parameters)

    # create autoregressive variables
    endog = ts[TEST_VALUE]
    deseas = endog.diff(YEAR_HRS).dropna()
    diff = deseas.diff(DAY_HRS).dropna()

    # if arimax, create exogonous
    if model.category == "arimax":
        exog = ts.drop(TEST_VALUE, axis=1)
        exog = fd.normalize(exog)
        if variance < 1:
            pca = PCA(variance)
    else:
        exog=None

    arima_predictions = []

    for day in dates.test.days:
        day = pd.date_range(start=day, periods=24, freq='h')
        start = len(endog.reindex(dates.train.hours).dropna())
        end = start + 23

        if variance < 1:
            exog_reduced = exog
            pca.fit(exog_reduced.reindex(dates.train.hours))
            exog_reduced = pd.DataFrame(pca.transform(exog_reduced),
                index=exog_reduced.index)

        arima_fit = arima(endog.reindex(dates.train.hours),
                exog_reduced.reindex(dates.train.hours)
                if exog_reduced is not None else exog_reduced)
        print(arima_fit.summary())
        arima_predictions.append(arima_fit.predict(start=start, end=end,
            exog=exog_reduced.reindex(dates.test.hours)
            if exog_reduced is not None else exog_reduced)
                + deseas.shift(DAY_HRS).loc[day]
                + endog.shift(YEAR_HRS).loc[day])

        dates.train.shift(pd.Timedelta("1 Day"))
        dates.test.shift(pd.Timedelta("1 Day"))

    arima_predictions = pd.concat(arima_predictions)
    test_results.yhat = arima_predictions

    if write is not None:
        model.to_excel(write + ".xlsx")
        test_results.plot(write + "_test.png", show=False)
        test_results.to_excel(write + ".xlsx")
        test_results.to_msgpack(write + "_test.msg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='data file')

    parser.add_argument('-m', '--model', help='model file')
    parser.add_argument('-s', '--span', help='date range')
    parser.add_argument('-v', '--variance', help='pca variance', default=1)
    parser.add_argument('-w', '--write', help='write results')

    args = parser.parse_args()

    main(args.data, args.model, args.span, float(args.variance), args.write)
