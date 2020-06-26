import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import itertools
from NNDataset import NNDataset
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

def to_day(ts):
    # separate date and time
    ts = ts.reset_index()
    ts['Hour'] = ts['Date'].dt.hour
    ts['Date'] = ts['Date'].dt.date

    # bug in pandas requires columns in sorted order
    cols = ts.columns.tolist()
    cols.sort()
    ts = ts[cols]

    # reset index
    ts = ts.set_index(['Date', 'Hour'])

    return ts

def get_channel(ts):
    # convert ts to daily data
    daily = to_day(ts)

    # normalize while like data is in columns
    daily = normalize(daily)

    # reset index
    channel_data = daily.reset_index()
    channel_data = channel_data.set_index('Date')

    # set columns as hours
    channel_data = channel_data.pivot(columns='Hour')

    # set data points as a third index level
    channel_data = channel_data.stack(level=0)

    # rename indeces
    channel_data.index = channel_data.index.set_names(['Date', 'Channel'])

    # rename columns
    channel_data.columns = pd.MultiIndex.from_product([['CURRENT'],
        [hr for hr in range(24)]])

    return channel_data

def ffnn_data(data, lag_days):
    sl_data = to_day(data)
    sl_data = normalize(sl_data)
    sl_data = add_lag(sl_data, lag_days)

    return sl_data

def cnn_data(data, lag_days):
    channel_data = get_channel(data)
    channel_data = add_lag(channel_data, lag_days)
    channel_data = channel_data.drop('CURRENT', axis=1, level=0)

    return channel_data

def reduce(data, var):
    pca = PCA(var)
    pca.fit(data)
    data = pd.DataFrame(pca.transform(data), index=data.index)

    return data

def add_lag(df, lag_days):
    groups = [name for name in df.index.names if name is not 'Date']

    # get columns to match existing column headers
    cols = [col[-1] if type(col) is tuple else col for col in df.columns.tolist()]

    # add lag days
    df_lag = pd.concat([df,
        *[df.groupby(by=groups).shift(lag) for lag in lag_days]], axis=1)

    # drop days with unknown values
    df_lag = df_lag.dropna()

    # rename columns
    df_lag.columns = pd.MultiIndex.from_product([['CURRENT',
        *['LAG ' + str(lag) for lag in lag_days]], [col for col in cols]])

    return df_lag

def remove_channel_rt(data):
    idx = pd.IndexSlice
    # mask values to change
    ch_mask = data.index.get_level_values('Channel').str.startswith('DA')
    ch_mask = [not channel for channel in ch_mask]

    # remove masked values
    data.loc[idx[:, :, ch_mask], 'CURRENT'] = 0

    return data

def create_dls(x, y, var, batch_size, test_value, dates):
    dls = {}
    dls['test'] = {}

    workers = os.cpu_count()
    y = y.unstack()

    x_train = x.loc[dates.train.days]
    x_val = x.loc[dates.validation.days]
    x_test = x.loc[dates.test.days]

    y_train = y.loc[dates.train.days]
    y_val = y.loc[dates.validation.days]
    y_test = y.loc[dates.test.days]

    pca = PCA(var)
    pca.fit(x_train)
    x_train = pd.DataFrame(pca.transform(x_train), index=x_train.index)
    x_val = pd.DataFrame(pca.transform(x_val), index=x_val.index)
    x_test = pd.DataFrame(pca.transform(x_test), index=x_test.index)

    dl_train = DataLoader(dataset=NNDataset(x_train, y_train),
            batch_size=batch_size, num_workers=workers, shuffle=False)

    dl_validation = DataLoader(dataset=NNDataset(x_val, y_val),
            batch_size=batch_size, num_workers=workers, shuffle=False)

    dl_test = DataLoader(dataset=NNDataset(x_test, y_test),
        batch_size=batch_size, num_workers=workers, shuffle=False)

    return dl_train, dl_validation, dl_test, x_train.shape[1]

def create_variable(v):
    # cast data to torch.autograd.Variable
    new_var = Variable(v, requires_grad=False).float()
    if torch.cuda.is_available():
        new_var = new_var.cuda()

    return new_var

def create_variables(data): 
    new_vars = []

    for  new_var in data:
        new_vars.append(create_variable(new_var))

    return tuple(new_vars)

def timeseries_index(x):
    x = x.reset_index()
    x['Date'] = x['Date'] + pd.to_timedelta(x['Hour'], unit='h')
    x = x.set_index('Date')
    x = x.drop(columns='Hour')

    return x

def normalize(df):
    return (df - df.mean()) / df.std()

def undifference(history, pred, lag=1):
    pred = pred + history.shift(-lag)
    return pred
