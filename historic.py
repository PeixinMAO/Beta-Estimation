from rollingbeta import RollingBeta
from statsmodels.regression.linear_model import OLS
import numpy as np

class RealizedBeta(RollingBeta):

    @staticmethod
    def realizedBeta(df):
        stockVol = np.std(df[:, 0],ddof=1)
        indexVol = np.std(df[:, 1],ddof=1)
        correl = np.corrcoef(df[:, 0],df[:, 1])[0][1]
        return correl * stockVol/indexVol

    def __init__(self, window, stock, index, ticker):
        super().__init__(window, stock, index, self.realizedBeta, ticker)


class RollingOLSBeta(RollingBeta):

    @staticmethod
    def ols(df):
        reg = OLS(df[:, 0], df[:, 1]).fit()
        res = reg.params[0]
        return res

    def __init__(self, window, stock, index, ticker):
        super().__init__(window, stock, index, self.ols, ticker)
