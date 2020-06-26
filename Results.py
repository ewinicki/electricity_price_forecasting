import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler
import msgpack
import os
from write_excel import write_excel
from error_calcs import rmse, mape

MONOCHROME = (cycler('color', ['k']) * cycler('marker', ['', '.', '^', 'o']) *
        cycler('linestyle', ['-', '--', ':', '-.']))

monochrome_simple = (cycler('color', ['k']) * cycler('linestyle',
    ['-', '--', ':', '-.']))

class TrainResults(object):
    def __init__(self):
        self._data = None

    def __init__(self, data):
        self._data = data

    def plot(self, write_path=None, show=True):
        rcParams.update({'figure.autolayout': True})
        plt.rc('axes', prop_cycle=MONOCHROME)
        ax = self.mean.plot(grid=True, linewidth=0.75)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Losses [$(\$/MWh)^2$]")

        if write_path is not None:
            plt.savefig(write_path)
            plt.close()

        if show:
            plt.show()

    def kde(self, write_path=None, show=True):
        plt.figure()
        summary = self.summary["epochs"]
        # if summary.max() > summary.min():
        #     bins = summary.max() - summary.min()
        # else:
        #     bins = 5

        plt.rc('axes', prop_cycle=MONOCHROME)
        # ax = summary.hist(grid=True, bins=bins)
        ax = summary.plot.kde(grid=True)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Density")

        if write_path is not None:
            plt.savefig(write_path)

        if show:
            plt.show()

    def add(self, new_sample):
        if self._data is None:
            self._data = new_sample
            self._data.columns = pd.MultiIndex.from_product([[0], new_sample.columns],
                    names=["sample", "data"])
            self._data.index.name = "epoch"
        else:
            sample = self._data.columns.get_level_values("sample").max() + 1 
            for col in new_sample.columns:
                self._data[sample, col] = new_sample[col]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def training(self):
        return self._data.loc[:, pd.IndexSlice[:, "training"]]

    @property
    def validation(self):
        return self._data.loc[:, pd.IndexSlice[:, "validation"]]

    @property
    def summary(self):
        summary = pd.DataFrame(index=range(self._data.columns.get_level_values("sample").max() + 1),
            columns=["epochs"])
        for sample in self.validation.columns.droplevel("data"):
            summary.loc[sample] = self.validation[sample].idxmin().values

        return summary

    @property
    def optimal(self):
        return round(self.summary["epochs"].mean())

    @property
    def optimal_median(self):
        return round(self.summary["epochs"].median())

    @property
    def optimal_mode(self):
        return round(self.summary["epochs"].mode().to_numpy()[0])

    @property
    def mean(self):
        _mean = pd.DataFrame(columns=["training", "validation"])
        _mean["training"] = self.training.mean(axis=1)
        _mean["validation"] = self.validation.mean(axis=1)
        return _mean

    def to_msgpack(self, path):
        self._data.to_msgpack(path)

    def __repr__(self):
        return self._data.__repr__()

    def __str__(self):
        return self._data.__str__()

    def write_optimal(self, fn):
        with open(fn, 'w') as epoch_file:
            epoch_file.write(str(self.optimal))

class TestResults(object):
    def __init__(self):
        self._data = pd.DataFrame()

    def __init__(self, data):
        self._data = data

    @property
    def yhat(self):
        return self._data["yhat"]

    @yhat.setter
    def yhat(self, yhat):
        self._data["yhat"] = yhat
        self._data["y - yhat"] = self._data["y"] - self._data["yhat"]

    @property
    def y(self):
        return self._data["y"]

    @y.setter
    def y(self, y):
        self._data["y"] = y

    @property
    def errors(self):
        errors = pd.DataFrame([], index=['RMSE', 'MAPE'],
                columns=self._data.columns)
        errors.loc['RMSE'] = rmse(self.yhat, self.y)
        errors.loc['MAPE'] = mape(self.yhat, self.y)

        return errors.drop("y", axis=1)

    @property
    def stats(self):
        return pd.DataFrame([self._data.min(), self._data.max(),
            self._data.mean(), self._data.median(), self._data.std()],
            index=['Min', 'Max', 'Mean', 'Median', 'Std Dev'])

    def __repr__(self):
        return self._data.__repr__()

    def __str__(self):
        return self._data.__str__()

    def to_msgpack(self, path):
        self._data.to_msgpack(path)

    def plot(self, write_path=None, show=True):
        rcParams.update({'figure.autolayout': True})
        plt.rc('axes', prop_cycle=MONOCHROME)
        ax = self._data.drop(["y - yhat"], axis=1).plot(grid=True, linewidth=0.75)
        ax.set_xlabel("Date")
        ax.set_ylabel("RT_LMP [$/MWh]")
        ax.legend(["actual", "predicted"])

        if write_path is not None:
            plt.savefig(write_path)
            plt.close()

        if show:
            plt.show()

    def to_excel(self, fn):
        write_excel(self.errors, fn, "errors")
        write_excel(self.stats, fn, "stats")
