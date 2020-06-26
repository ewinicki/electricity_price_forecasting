import json
import pandas as pd

class DateRange(object):
    def __init__(self, start, end):
        self._start = pd.Timestamp(start)
        self._end = pd.Timestamp(end)

    def __init__(self, date_dict):
        self._start = pd.Timestamp(date_dict["start"])
        self._end = pd.Timestamp(date_dict["end"])

    def __call__(self):
        return pd.date_range(self.start, self.end)

    def shift(self, time):
        self.start += time
        self.end += time

    @property
    def hours(self):
        return pd.date_range(self.start, self.end, freq='H', closed='left')

    @property
    def days(self):
        return pd.date_range(self.start, self.end, freq='D', closed='left')

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, start):
        self._start = pd.Timestamp(start)

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, end):
        self._end = pd.Timestamp(end)

class DateRanges(dict):
    def __init__(self, dates_json):
        with open(dates_json, 'r') as dates_file:
            date_ranges = json.load(dates_file)

        for name, date_range in date_ranges.items():
            self[name] = DateRange(date_range)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError("{} does not exist".format(key))

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError("{} does not exist".format(key))
