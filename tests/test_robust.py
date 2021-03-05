# %%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/../")))

from robust import WinsorizedBeta, QuantileBeta, RANSACBeta, KalmanBeta, WelchBeta
from load_data import logReturns

ticker = "EKT SM Equity"
index = "SX5E Index"

a = WinsorizedBeta(63,logReturns.data[ticker], 
                logReturns.data[index], 
                ticker,2)
assert a.calc().shape[0] == 2714

a = QuantileBeta(63,
                logReturns.data[ticker], 
                logReturns.data[index], 
                ticker)

assert a.calc().shape[0] == 2714

a = RANSACBeta(63,
                logReturns.data[ticker], 
                logReturns.data[index], 
                ticker)

assert a.calc().shape[0] == 2714

a = KalmanBeta(logReturns.data[ticker], 
                logReturns.data[index], 
                ticker)
assert a.calc().shape[0] == 2714


a = WelchBeta(210, 2/252,logReturns.data[ticker], 
                logReturns.data[index],
                ticker)
assert a.calc().shape[0] == 2714


# %%
