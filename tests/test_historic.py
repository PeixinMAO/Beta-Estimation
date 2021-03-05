import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__+"/../")))

from historic import RealizedBeta, RollingOLSBeta
from load_data import logReturns

ticker = "AC FP Equity"
index = "SX5E Index"
a = RealizedBeta(63,
                logReturns.data[ticker], 
                logReturns.data[index], 
                ticker)

assert a.calc().shape[0] == 2714

a = RollingOLSBeta(63,
                logReturns.data[ticker], 
                logReturns.data[index], 
                ticker)

assert a.calc().shape[0] == 2714