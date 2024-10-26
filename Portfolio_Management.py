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

Inputs

'''

risk_free_rate = 0.04 # change the input to current Treasury Yield

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days = 365)

portfolio = ['^GSPC', 'TSLA', 'NVDA'] # example with S&P500, TESLA, NVIDA
weights = np.array([1/len(portfolio), 1/len(portfolio), 1/len(portfolio)])


'''

Outputs

'''

meanReturns, covarianceMatrix = getData(portfolio, start = startDate, end = endDate)
returns, standard_deviation = portfolioPerformance(weights, meanReturns, covarianceMatrix)

# result = (maxSR(meanReturns, covarianceMatrix, risk_free_rate))
# maxSharpe, optimal_weights = result['fun'], result['x']
# print(maxSharpe, optimal_weights)

result = minimizeVariance(meanReturns, covarianceMatrix)
mininumVariance, optimal_weights = result['fun'], result['x']
print(mininumVariance, optimal_weights)