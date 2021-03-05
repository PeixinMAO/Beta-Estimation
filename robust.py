from rollingbeta import RollingBeta
import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.regression.linear_model import OLS, WLS
import sklearn.linear_model
import statsmodels.api as sm
from pykalman import KalmanFilter
import scipy as sp

class WinsorizedBeta(RollingBeta):

    def winsorizedOLS(self, df):
        if sum(np.isnan(df[:, 0])) == self.window:
            return "NaN"
        if sum(np.isnan(df[:, 0])) > 0:
            df[:, 0] = df[:, 0][~np.isnan(df[:, 0])]
        df[:, 0] = sp.stats.mstats.winsorize(df[:, 0], self.cutoff, axis=0).data
        reg = OLS(df[:, 0], df[:, 1]).fit()
        res = reg.params[0]
        return res

    def __init__(self, window, stock, index, ticker, nboutliers):
        super().__init__(window, stock, index, self.winsorizedOLS, ticker)
        self.cutoff = (np.ceil(100*(nboutliers/self.window))/100)


class QuantileBeta(RollingBeta):

    @staticmethod
    def quantileReg(df):
        reg = QuantReg(df[:, 0], df[:, 1]).fit(q=.5)
        res = reg.params[0]
        return res

    def __init__(self, window, stock, index, ticker):
        super().__init__(window, stock, index, self.quantileReg, ticker)


class RANSACBeta(RollingBeta):

    @staticmethod
    def ransacReg(df):
        X = df[:, 1].reshape(-1, 1)
        Y = df[:, 0][~np.isnan(df[:, 0]).any(
            axis=0)].reshape(-1, 1)
        if (Y.shape[0] != X.shape[0]) or (Y.shape[0] == 0) or (X.shape[0] == 0):
            return 'NaN'
        try:
            reg = sklearn.linear_model.RANSACRegressor(random_state=0).fit(X, Y)
            res = reg.estimator_.coef_[0]
            return res[0]
        except ValueError:
            return 'NaN'

    def __init__(self, window, stock, index, ticker):
        super().__init__(window, stock, index, self.ransacReg, ticker)


class KalmanBeta(object):

    def __init__(self, stock, index, ticker, n_iter=2):
        self.df = pd.DataFrame(
            {ticker: stock, "index": index}).dropna()
        self.X = sm.add_constant(self.df["index"])
        self.Y = self.df[ticker]
        self.ticker = ticker
        self.n_iter = n_iter

    def calc(self):
        kf = KalmanFilter(n_dim_obs=1,
                          n_dim_state=2,
                          transition_matrices=np.eye(2),
                          observation_matrices=np.expand_dims(
                              self.X, axis=1),
                          em_vars=["transition_covariance",
                                   "observation_covariance",
                                   "initial_state_covariance",
                                   "initial_state_mean"])
        kf = kf.em(self.Y.values, n_iter=self.n_iter)
        results = kf.filter(self.Y.values)
        kalman_factors = pd.DataFrame(results[0], columns=[
                                      "alpha", self.ticker], index=self.df.index).drop(columns="alpha")
        return kalman_factors


class WelchBeta(RollingBeta):

    def wls(self, df):
        retlow = np.min(np.vstack(((-2) * df[:, 1], 4 * df[:, 1])), axis=0)
        rethigh = np.max(np.vstack(((-2) * df[:, 1], 4 * df[:, 1])), axis=0)
        ret = np.min(np.vstack((np.max(np.vstack((retlow, df[:, 0])), axis=0), rethigh)), axis=0)
        weights = np.array([np.exp(-self.rho*n) for n in range(df[:,0].shape[0])])
        reg = WLS(ret, df[:, 1], weights=weights).fit()
        res = reg.params[0]
        return res

    def __init__(self, window, rho, stock, index, ticker):
        super().__init__(window, stock, index, self.wls, ticker)
        self.rho = rho
