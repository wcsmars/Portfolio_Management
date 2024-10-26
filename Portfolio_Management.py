import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import scipy.optimize as sc

def getData(portfolio, start, end):
    stockData = yf.download(portfolio, start = start, end = end)
    stockDatas = stockData['Close']
    returns = stockDatas.pct_change().dropna()
    meanReturns = returns.mean()
    covarianceMatrix = returns.cov()
    return meanReturns, covarianceMatrix
    #return (stockDatas[-1] - stockDatas[0]) / stockDatas[0] # 1-year return

def portfolioPerformance(weights, meanReturns, covarianceMatrix):
    returns = np.sum(weights * meanReturns) * 252 # needa times 252 because the meanReturns are daily.
    portfolio_variance = np.dot(weights.T, np.dot(covarianceMatrix, weights))
    standard_deviation = np.sqrt(portfolio_variance) * np.sqrt(252)
    return returns, standard_deviation

'''

Maximize Sharpe Ratio

'''

def negativeSR(weights, meanReturns, covarianceMatrix, risk_free_rate):
    portfolioReturn, portfolioSTD = portfolioPerformance(weights, meanReturns, covarianceMatrix)
    return - (portfolioReturn - risk_free_rate) / portfolioSTD # negative sharpe ratio here because I use the minimize function to maximize the postive sharp ratio

def maxSR(meanReturns, covarianceMatrix, risk_free_rate):
    num_of_assets = len(portfolio)
    args = (meanReturns, covarianceMatrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_of_assets))
    result = sc.minimize(negativeSR, num_of_assets * [1. / num_of_assets], args = args,
                         method = 'SLSQP', bounds = bounds, constraints = constraints)
    return result

'''

Minimize Portfolio Variance

'''

def portfolioVariance(weights, meanReturns, covarianceMatrix):
    return portfolioPerformance(weights, meanReturns, covarianceMatrix)[1]

def minimizeVariance(meanReturns, covarianceMatrix):
    num_of_assets = len(portfolio)
    args = (meanReturns, covarianceMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_of_assets))
    result = sc.minimize(portfolioVariance, num_of_assets * [1. / num_of_assets], args = args,
                         method = 'SLSQP', bounds = bounds, constraints = constraints)
    return result

'''

Output result function

'''

def sharpe_outputs(meanReturns, risk_free_rate, covarianceMatrix):
    maxSharpe = maxSR(meanReturns, covarianceMatrix, risk_free_rate)
    maxSharpe_Return, maxSharpe_STD = portfolioPerformance(maxSharpe['x'], meanReturns, covarianceMatrix)
    maxSharpe_Allocation = pd.DataFrame(maxSharpe['x'], index = meanReturns.index, columns = ['Weights'])
    maxSharpe_Allocation.Weights = [round(i * 100, 0) for i in maxSharpe_Allocation.Weights]
    return maxSharpe_Return, maxSharpe_STD, maxSharpe_Allocation

def minimumVariance_outputs(meanReturns, covarianceMatrix):
    minimumVariance = minimizeVariance(meanReturns, covarianceMatrix)
    minimumVariance_Return, minimumVariance_STD = portfolioPerformance(minimumVariance['x'], meanReturns, covarianceMatrix)
    minimumVariance_Allocation = pd.DataFrame(minimumVariance['x'], index = meanReturns.index, columns = ['Weights'])
    minimumVariance_Allocation.Weights = [round(i * 100, 0) for i in minimumVariance_Allocation.Weights]
    return minimumVariance_Return, minimumVariance_STD, minimumVariance_Allocation


'''

Inputs

'''

risk_free_rate = 0.04 # change the input to current Treasury Yield

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 365)

portfolio = ['^GSPC', '^HSI'] # input tickers
weights = np.array([1/len(portfolio), 1/len(portfolio)]) # adjustment needed, use for-loop instead


'''

Outputs

'''

meanReturns, covarianceMatrix = getData(portfolio, start = startDate, end = endDate)
returns, standard_deviation = portfolioPerformance(weights, meanReturns, covarianceMatrix)

# result = (maxSR(meanReturns, covarianceMatrix, risk_free_rate))
# maxSharpe, optimal_weights = result['fun'], result['x']
# print(maxSharpe, optimal_weights)

# result = minimizeVariance(meanReturns, covarianceMatrix)
# mininumVariance, optimal_weights = result['fun'], result['x']
# print(mininumVariance, optimal_weights)

# print(sharpe_outputs(meanReturns, risk_free_rate, covarianceMatrix))

print(minimumVariance_outputs(meanReturns, covarianceMatrix))