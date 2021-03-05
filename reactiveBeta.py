# %%
# Add the current directory into path so we can import our functions
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import matplotlib.pyplot as plt

# Require Dev verson of statsmodels 0.11
# pip install git+https://github.com/statsmodels/statsmodels.git@v0.11.0dev0 --upgrade


# %%
histoStocks = pd.read_csv(
    f'{os.path.abspath(__file__ + "/../../data")}/candlestick_fund_And_SX5E.csv')
histoStocks["Date"] = pd.to_datetime(
    histoStocks["Date"], format='%d/%m/%Y')


def extractPrices(histoStocks, ptype):
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


dfpOpen = extractPrices(histoStocks, "Open")
dfpHigh = extractPrices(histoStocks, "Low")
dfpLow = extractPrices(histoStocks, "High")
dfpLast = extractPrices(histoStocks, "Last")
# %%
LAMBDA_F = 0.484
LAMBDA_S = 0.0241

I = dfpLast["SX5E Index"]
S_i = dfpLast.drop("SX5E Index", axis=1)

L_f = I.ewm(alpha=LAMBDA_F, adjust=False).mean()
L_s = I.ewm(alpha=LAMBDA_S, adjust=False).mean()
L_is = S_i.ewm(alpha=LAMBDA_S, adjust=False).mean()
# %%
L_INDEX_EXP_FIT_PARAM = 8
L_STOCK_EXP_FIT_PARAM = 7.09
PHI = 3.3
L = I*(1+(PHI*(L_s-I)/I).apply(np.tanh))*(1+L_INDEX_EXP_FIT_PARAM*(L_f-I)/L_f)
L_i = S_i.multiply(1+(PHI*L_is.sub(S_i, axis=0).div(S_i, axis=0)).apply(np.tanh),
                   axis=0).multiply(1+L_STOCK_EXP_FIT_PARAM*(L_f-I)/L_f, axis=0)
# %%
return_stock = S_i.sub(S_i.shift(), axis=0).div(
    S_i.shift(), axis=0).dropna(how='all')
return_index = I.sub(I.shift()).div(I.shift()).dropna()
normalized_return_index = I.sub(
    I.shift()).div(L.shift()).dropna()
normalized_return_stock = S_i.sub(S_i.shift(), axis=0).div(
    L_i.shift(), axis=0).dropna(how='all')
# %%
VOL_WEIGHT_PARAM = 1/40
normalized_vol_index = np.sqrt(np.square(normalized_return_index).ewm(
    alpha=VOL_WEIGHT_PARAM, adjust=False).mean())
normalized_vol_stock = np.sqrt(np.square(normalized_return_stock).ewm(
    alpha=VOL_WEIGHT_PARAM, adjust=False).mean())
# %%
reactive_vol_index = normalized_vol_index * L/I
reactive_vol_stock = normalized_vol_stock.multiply(
    L_i.div(S_i, axis=0), axis=0)
# %%
return_stock = S_i.sub(S_i.shift(), axis=0).div(
    S_i.shift(), axis=0)
return_index = (I - I.shift())/I.shift()

ols_beta = pd.DataFrame()

for stock in S_i.columns:
    model = RollingOLS(
        return_stock[stock], return_index, window=63)
    results = model.fit()
    ols_beta[stock] = results.params.values.flatten()
ols_beta = ols_beta.set_index(
    return_stock.index).dropna(how='all')
# %%
normalized_ols_beta = pd.DataFrame()

for stock in S_i.columns:
    model = RollingOLS(
        normalized_return_stock[stock], normalized_return_index, window=63)
    results = model.fit()
    normalized_ols_beta[stock] = results.params.values.flatten(
    )
normalized_ols_beta = normalized_ols_beta.set_index(
    normalized_return_stock.index).dropna(how='all')

# %%
# reactive_beta = normalized_ols_beta.multiply(
#     L_i.multiply(
#         I,axis=0).div(
#             S_i.multiply(
#                 L,axis=0),
#                 axis=0),
#                 axis=0).dropna(how="all")
# # Alternatively
# reactive_beta = normalized_ols_beta.multiply(
#     reactive_vol_stock.multiply(
#         normalized_vol_index,axis=0).div(
#             normalized_vol_stock.multiply(
#                 reactive_vol_index,axis=0),
#                 axis=0),
#                 axis=0).dropna(how="all")

# %%
renormalized_return_index = normalized_return_index.div(
    normalized_vol_index.shift()).dropna()
renormalized_return_stock = normalized_return_stock.div(
    normalized_vol_stock.shift(), axis=0).dropna(how='all')

# %%
LAMBDA_BETA = 1/90
renormalized_vol_index = np.sqrt(np.square(renormalized_return_index).ewm(
    alpha=VOL_WEIGHT_PARAM, adjust=False).mean())
renormalized_vol_stock = np.sqrt(np.square(renormalized_return_stock).ewm(
    alpha=VOL_WEIGHT_PARAM, adjust=False).mean())

phi_hat_i = renormalized_return_stock.multiply(
    renormalized_return_index, axis=0).ewm(alpha=VOL_WEIGHT_PARAM, adjust=False).mean()

renormalized_beta = phi_hat_i.div(
    np.square(renormalized_vol_index), axis=0)

# %%
# Reproduce figure 3
x = renormalized_beta - renormalized_beta.mean()
x = x[(x.index >= "2014") & (
    x.index <= "2015")].dropna(axis=1)
y = np.log(renormalized_vol_stock.div(renormalized_vol_index, axis=0)).sub(
    np.log(renormalized_vol_stock.div(renormalized_vol_index, axis=0).mean()))
y = y[(y.index >= "2014") & (
    y.index <= "2015")].dropna(axis=1)
# %%
plt.scatter(y.values.flatten(), x.values.flatten())
# Strange exponential curve
# %%
# Strange curve due to this particular stock
plt.scatter(y[['BNR GY Equity']].values.flatten(),
            x[['BNR GY Equity']].values.flatten())
# %%
plt.plot(dfpLast['BNR GY Equity'])

# %%


def f(normalized_beta):
    if normalized_beta <= .5:
        return 0
    elif (normalized_beta > 0.5) & (normalized_beta <= 1.6):
        return 0.6*(normalized_beta-.5)
    else:
        return 0.6


# %%
kappa_i = np.square(normalized_vol_stock.div(normalized_vol_index, axis=0)).ewm(
    alpha=LAMBDA_BETA, adjust=False).mean()


def delta(relative_normalized_vol):
    return relative_normalized_vol.shift().sub(np.sqrt(kappa_i.shift()), axis=0).div(np.sqrt(kappa_i.shift()), axis=0)


F = 1 + 2*renormalized_beta.applymap(f).div(renormalized_beta, axis=0).multiply(
    delta(normalized_vol_stock.div(normalized_vol_index, axis=0)), axis=0).dropna(how="all")

# %%
# correlData = pd.read_csv(
#     f'{os.path.abspath(__file__ + "/../../data")}/SX5E Correls.csv',
#     skiprows=range(7),
#     usecols=(1, 2, 5, 8, 11, 14, 17),
#     names=["Date", "IC_3M", "IC_6M", "IC_12M", "HC_3M", "HC_6M", "HC_12M"])

# correlData["Date"] = pd.to_datetime(correlData["Date"], format='%m/%d/%Y')
# correlData = correlData.dropna()  # Remove missing days
# correlData = correlData.set_index('Date')
# correlData = correlData.reindex(index=correlData.index[::-1])
# y=correlData["IC_3M"][(correlData["IC_3M"].index>="2015-01-02")&(correlData["IC_3M"].index<"2016")]
# y = y - y.mean()
# x=((L_f-I)/L_f)
# x=x[(x.index>="2015")&(x.index<="2016")]
# plt.scatter(x,y)
# %%
L_INDEX_EXP_FIT_PARAM = 8
L_STOCK_EXP_FIT_PARAM = 7.09

correction_factor = 1+(L_INDEX_EXP_FIT_PARAM-L_STOCK_EXP_FIT_PARAM) * \
    (L_f.shift().sub(I.shift()).div(L_f.shift())).dropna()

# %%
normalized_cov_of_normalized_returns = renormalized_return_stock.multiply(renormalized_return_index, axis=0).div(
    F.multiply(correction_factor, axis=0)).ewm(alpha=LAMBDA_BETA, adjust=False).mean().dropna(how="all")

normalized_beta = normalized_cov_of_normalized_returns.div(
    np.square(renormalized_vol_index), axis=0).dropna(how="all")
# %%
reactive_beta_estimate = normalized_beta.multiply(
    L_i.multiply(
        I, axis=0).div(
            S_i.multiply(
                L, axis=0),
        axis=0),
    axis=0).multiply(F.multiply(correction_factor, axis=0), axis=0).dropna(how="all")
reactive_beta_estimate[reactive_beta_estimate.index>="2016-06-01"].to_csv(
    f'{os.path.abspath(__file__ + "/../../output")}/reactive_beta_estimate_3M.csv')

# %%



### Note à Khalil:
### ligne 55,56, l'un des deux vaux plutôt 7.
### ligne 207, c'est plutôt renormatized retrun au lieu de normalized return (ie r chapeau et non r tilde)
### J'ai pas vu la note la dernière fois. Il faut en faite appliquer un filtre dans L et L_i. 
### (cf pied de la page 6 de l'article).
### Je ne comprend pas comment est construit la fontion f (ie le ols_beta), tu me dira cet aprem :)
### Est ce que tu peux préparer quelques plot du beta reactive avec le beta_realised_3M 
### (ie les deux sur le même plot) car ça sera plus facile de discuter dessus.
### Les véritables paramètres du modèles sont seulement l-l'(plutôt l' seulement) et f (les auteurs les fits), 
### les autres étant plus ou moins fixes donc il serait sympa de voir l'évolution des courbes en fonction de 
### ces paramètres car on aura une idée sur où est situé le problème (Implémentation? Mauvaises paramètres?)