#!/usr/bin/python3

import argparse
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

from Results import TestResults
from Results import TrainResults

import matplotlib.pyplot as plt

def read_files(results):
    with open(results, 'r') as data_file:
        files = data_file.readlines()

    for line in files:
        line = line.rstrip()
        if "train" in line:
            data = TrainResults(pq.read_table(line).to_pandas())
        elif "test" in line:
            data = TestResults(pq.read_table(line).to_pandas())

        data.plot(write_path=line.rsplit('.', 1)[0]+'.png', show=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results')
    args = parser.parse_args()

    read_files(args.results)
