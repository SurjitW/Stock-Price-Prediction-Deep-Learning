from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas_datareader.data as pdr_data
import numpy as np
import time
import os
import sys
from collections import deque

import tensorflow as tf
from tensorflow.python.ops import rnn

import tfs_config as c
import csv

"""
Adapted from Google's PTB word prediction TensorFlow tutorial.

Copyright 2016 Tencia Lee

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

class DownloadSymbolPrices(object):

    '''
    If filename exists, loads data, otherwise downloads and saves data
    from Yahoo Finance
    Returns:
    - a list of arrays of close-to-close percentage returns, normalized by running
      stdev calculated over last c.normalize_std_len days
    '''
    def download_data(self):
        from datetime import timedelta, datetime
        # find date range for the split train, val, test (0.8, 0.1, 0.1 of total days)
        
        split = [0.8, 0.1, 0.1]
        cumusplit = [np.sum(split[:i]) for i,s in enumerate(split)]
        segment_start_dates = [c.start + timedelta(
            days = int((c.end - c.start).days * interv)) for interv in cumusplit][::-1]
        stocks_list = [c.symbol]
        print('Downloading data for dates {} - {}'.format(
            datetime.strftime(c.start, "%Y-%m-%d"),
            datetime.strftime(c.end, "%Y-%m-%d")))
        aapl = pdr_data.DataReader(c.symbol, 'yahoo', c.start, c.end)
        by_stock = dict((s, pdr_data.DataReader(c.symbol, 'yahoo', c.start, c.end))
                for s in stocks_list)
        aapl.to_csv(sys.path[0] + '\\data\\' + c.symbol + '_full.csv')
        print('Downloaded data for dates {} - {}'.format(
            datetime.strftime(c.start, "%Y-%m-%d"),
            datetime.strftime(c.end, "%Y-%m-%d")))
        seq = [[],[],[]]
        lastAct = []
        for stock in by_stock:
            lastAc = -1
            daily_returns = deque(maxlen=c.normalize_std_len)
            for rec_date in (c.start + timedelta(days=n) for n in range((c.end-c.start).days)):
                idx = next(i for i,d in enumerate(segment_start_dates) if rec_date >= d)
                try:
                    d = rec_date.strftime("%Y-%m-%d")
                    ac = by_stock[stock].ix[d]['Adj Close']
                    lastAct.append(ac)
                    daily_return = (ac - lastAc)/lastAc
                    if len(daily_returns) == daily_returns.maxlen:
                        seq[idx].append(daily_return/np.std(daily_returns))
                    daily_returns.append(daily_return)
                    lastAc = ac
                except KeyError:
                    pass
        print(lastAct)
        return lastAct
    

    def save_data(self):
        file = sys.path[0] + '\\' + c.save_file
        if not os.path.exists(file):
            print('Saving in {}'.format(file))
            datasets = self.download_data()
            print(datasets)
            with open(file, "wb") as f:
                w = csv.writer(f)
                w.writerows(datasets)
        else:
            with np.load(file) as file_load:
                datasets = [file_load['arr_%d' % i] for i in range(len(file_load.files))]
        return datasets

def main():
    d = DownloadSymbolPrices()
    d.download_data()
    print(d)
    
    #d.save_data()

if __name__ == "__main__":
    main()
