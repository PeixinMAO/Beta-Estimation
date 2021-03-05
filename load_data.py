import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

class logReturns:

    histoStocks = pd.read_csv(
        f'{os.path.abspath(__file__ + "/../../data")}/candlestick_fund_And_SX5E.csv')
    histoStocks["Date"] = pd.to_datetime(
        histoStocks["Date"], format='%d/%m/%Y')


    def _extractPrices(histoStocks, ptype):
        # We'll only use the last price for calculations at this point.
        histoData = histoStocks[[
            "Date"]+[colname for colname in histoStocks.columns if ptype in colname]]
        histoData.columns = [colname.replace(f".{ptype}", "")
                            for colname in histoData.columns]

        histoData = histoData.set_index('Date')
        # Drop rows with all NAs (eliminates holidays if ALL companies are traded in the same market)
        histoData = histoData.dropna(how='all')

        # Use the previous day price for missing days
        histoData = histoData.fillna(method='ffill')
        return histoData


    dfpLast = _extractPrices(histoStocks, "Last")

    def _calReturns(histoData, frequency="daily"):
        if (frequency == "daily"):
            dailyReturns = np.log(histoData) - np.log(histoData.shift(1))
            return dailyReturns.dropna(how='all')
        elif (frequency == "weekly"):
            weeklyReturns = np.log(histoData) - np.log(histoData.shift(5))
            return weeklyReturns.dropna(how='all')
        else:
            raise Exception("Unknown frequency")

    data =  _calReturns(dfpLast, frequency="daily")