import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided as strided

class RollingBeta:

    def __init__(self, window, stock, index, estimator, ticker):
        self.window = window
        self.ticker = ticker
        self.df = pd.DataFrame({ticker:stock, "index":index})
        self.estimator = estimator

    def _sliding_windows(self):
        v = self.df.values
        d0, d1 = v.shape
        s0, s1 = v.strides
        a = strided(v, shape=(d0 - (self.window - 1), self.window, d1), strides=(s0, s0, s1))
        return a

    def calc(self):
        windows = self._sliding_windows()
        result = np.array(list(map(self.estimator, windows)))
        return pd.DataFrame({self.ticker:result}, index = self.df.index[self.window-1:])