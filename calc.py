pip install pandas
# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed

from historic import RealizedBeta, RollingOLSBeta
from robust import WinsorizedBeta, QuantileBeta, RANSACBeta, KalmanBeta, WelchBeta
from load_data import logReturns

from statsmodels.regression.linear_model import OLS
from scipy.stats import kurtosis, probplot


def parallelCalcEstimator(Estimator, **kwargs):
    def calcEstimator(ticker):
        index = "SX5E Index"
        inst = Estimator(stock = logReturns.data[ticker], 
                        index = logReturns.data[index], 
                        ticker = ticker,
                        **kwargs)
        return inst.calc()
    result = Parallel(n_jobs=5)(delayed(calcEstimator)(ticker)
                                for ticker in logReturns.data.columns)

    return pd.concat(result, copy=False, axis=1)

# realizedBeta = parallelCalcEstimator(RealizedBeta, window=63)
realizedBeta = pd.read_csv(
    f'{os.path.abspath(__file__ + "/../../output")}/realizedBeta.csv')
realizedBeta["Date"] = pd.to_datetime(realizedBeta["Date"], format='%Y-%m-%d')
realizedBeta = realizedBeta.set_index('Date')

# rollingOLSBeta = parallelCalcEstimator(RollingOLSBeta, window=63)
rollingOLSBeta = pd.read_csv(
    f'{os.path.abspath(__file__ + "/../../output")}/rollingOLSBeta.csv')
rollingOLSBeta["Date"] = pd.to_datetime(rollingOLSBeta["Date"], format='%Y-%m-%d')
rollingOLSBeta = rollingOLSBeta.set_index('Date')

# winsorizedBeta = parallelCalcEstimator(WinsorizedBeta, window=63, nboutliers=2)
winsorizedBeta = pd.read_csv(
    f'{os.path.abspath(__file__ + "/../../output")}/winsorizedBeta.csv')
winsorizedBeta["Date"] = pd.to_datetime(winsorizedBeta["Date"], format='%Y-%m-%d')
winsorizedBeta = winsorizedBeta.set_index('Date')

# quantileBeta = parallelCalcEstimator(QuantileBeta, window=63)
quantileBeta = pd.read_csv(
    f'{os.path.abspath(__file__ + "/../../output")}/quantileBeta.csv')
quantileBeta["Date"] = pd.to_datetime(quantileBeta["Date"], format='%Y-%m-%d')
quantileBeta = quantileBeta.set_index('Date')

# ransacBeta = parallelCalcEstimator(RANSACBeta, window=63)
ransacBeta = pd.read_csv(
    f'{os.path.abspath(__file__ + "/../../output")}/ransacBeta.csv')
ransacBeta["Date"] = pd.to_datetime(ransacBeta["Date"], format='%Y-%m-%d')
ransacBeta = ransacBeta.set_index('Date')

# kalmanBeta = parallelCalcEstimator(KalmanBeta, n_iter=2)
kalmanBeta = pd.read_csv(
    f'{os.path.abspath(__file__ + "/../../output")}/kalmanBeta.csv')
kalmanBeta["Date"] = pd.to_datetime(kalmanBeta["Date"], format='%Y-%m-%d')
kalmanBeta = kalmanBeta.set_index('Date')

# welchBeta = parallelCalcEstimator(WelchBeta, window=210, rho=2/252)
welchBeta = pd.read_csv(
    f'{os.path.abspath(__file__ + "/../../output")}/welchBeta.csv')
welchBeta["Date"] = pd.to_datetime(welchBeta["Date"], format='%Y-%m-%d')
welchBeta = welchBeta.set_index('Date')
# %%
estimators = {
    "rollingOLSBeta":rollingOLSBeta,
    "winsorizedBeta":winsorizedBeta, 
    "quantileBeta":quantileBeta, 
    "ransacBeta":ransacBeta, 
    "kalmanBeta":kalmanBeta,
    "welchBeta":welchBeta
}
residuals = {}
for estimatedBeta in estimators.keys():
    residuals[estimatedBeta] = (estimators[estimatedBeta]-realizedBeta).dropna(how="all")

scaledResiduals = deepcopy(residuals)
for res in scaledResiduals.values():
    for ticker in res.columns:
        res[ticker] = (res[ticker] - np.mean(res[ticker])) / np.std(res[ticker],ddof=1)

# %%
rmseList = [(res**2).mean()**0.5 for res in residuals.values()]
metric_rmse = pd.concat(rmseList,axis=1)
metric_rmse.columns = estimators.keys()
metric_rmse = metric_rmse.drop("SX5E Index")
# %%
metric_rsquared = pd.DataFrame(index=realizedBeta.columns)
for estimatedBeta in estimators.keys():
    rsq = []
    for ticker in estimators[estimatedBeta].columns:
        df = pd.DataFrame({"estimated":estimators[estimatedBeta][ticker], "ground truth":realizedBeta[ticker]}).dropna()
        reg = OLS(df["estimated"], df["ground truth"]).fit()
        rsq.append(reg.rsquared)
    metric_rsquared[estimatedBeta] = rsq
metric_rsquared = metric_rsquared.drop("SX5E Index")
# %% 
kurtosisList = [pd.Series(kurtosis(res, nan_policy="omit").data, index=res.columns) for res in scaledResiduals.values()]
metric_kurtosis = pd.concat(kurtosisList, axis=1)
metric_kurtosis.columns = estimators.keys()
metric_kurtosis = metric_kurtosis.drop("SX5E Index")

# %%
metric_surface = pd.DataFrame(index=realizedBeta.columns)
for estimatedBeta in estimators.keys():
    surf = []
    for ticker in estimators[estimatedBeta].columns:
        quantiles=probplot(scaledResiduals[estimatedBeta][ticker], dist="norm",fit=False , plot=None)
        s = np.nansum(np.abs(quantiles[1] - quantiles[0])[:-1]*np.diff(quantiles[0]))
        surf.append(s)
    metric_surface[estimatedBeta] = surf
metric_surface = metric_surface.drop("SX5E Index")
# %%
metrics = {
    "RMSE": metric_rmse,
    "RSQ": metric_rsquared,
    "KURT": metric_kurtosis,
    "SURF": metric_surface
}

results = pd.concat([metric.mean() for metric in metrics.values()],axis=1)
results.columns = metrics.keys()

# %%
