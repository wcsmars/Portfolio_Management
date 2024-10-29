import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import scipy.optimize as sc
import plotly.graph_objects as go

def getData(portfolio, start, end):
    stockData = yf.download(portfolio, start = start, end = end)
    stockDatas = stockData['Close'].ffill()
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

Efficient optimization

'''


def portfolioReturn(weights, meanReturns, covarianceMatrix):
    return portfolioPerformance(weights, meanReturns, covarianceMatrix)[0]

def efficient_optimization(meanReturns, covarianceMatrix, returnTarget):
    num_of_assets = len(portfolio)
    args = (meanReturns, covarianceMatrix)
    bounds = tuple((0,1) for asset in range(num_of_assets))


    constraints = ({'type': 'eq', 'fun': lambda x: portfolioReturn(x, meanReturns, covarianceMatrix) - returnTarget},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    

    efficient_optimization_result = sc.minimize(portfolioVariance, num_of_assets * [1. / num_of_assets], args = args,
                                                method = 'SLSQP', bounds = bounds, constraints = constraints)
    
    return efficient_optimization_result


'''

Output result function

'''

# def sharpe_outputs(meanReturns, risk_free_rate, covarianceMatrix):
#     maxSharpe = maxSR(meanReturns, covarianceMatrix, risk_free_rate)
#     maxSharpe_Return, maxSharpe_STD = portfolioPerformance(maxSharpe['x'], meanReturns, covarianceMatrix)
#     maxSharpe_Allocation = pd.DataFrame(maxSharpe['x'], index = meanReturns.index, columns = ['Weights'])
#     maxSharpe_Allocation.Weights = [round(i * 100, 0) for i in maxSharpe_Allocation.Weights]
#     return maxSharpe_Return, maxSharpe_STD, maxSharpe_Allocation

# def minimumVariance_outputs(meanReturns, covarianceMatrix):
#     minimumVariance = minimizeVariance(meanReturns, covarianceMatrix)
#     minimumVariance_Return, minimumVariance_STD = portfolioPerformance(minimumVariance['x'], meanReturns, covarianceMatrix)
#     minimumVariance_Allocation = pd.DataFrame(minimumVariance['x'], index = meanReturns.index, columns = ['Weights'])
#     minimumVariance_Allocation.Weights = [round(i * 100, 0) for i in minimumVariance_Allocation.Weights]
#     return minimumVariance_Return, minimumVariance_STD, minimumVariance_Allocation


def results(meanReturns, covarianceMatrix, risk_free_rate):
    maxSharpe = maxSR(meanReturns, covarianceMatrix, risk_free_rate)
    maxSharpe_Return, maxSharpe_STD = portfolioPerformance(maxSharpe['x'], meanReturns, covarianceMatrix)
    maxSharpe_Allocation = pd.DataFrame(maxSharpe['x'], index = meanReturns.index, columns = ['Weights'])
    

    minimumVariance = minimizeVariance(meanReturns, covarianceMatrix)
    minimumVariance_Return, minimumVariance_STD = portfolioPerformance(minimumVariance['x'], meanReturns, covarianceMatrix)
    minimumVariance_Allocation = pd.DataFrame(minimumVariance['x'], index = meanReturns.index, columns = ['Weights'])
    

    efficient_list = []
    targetReturns = np.linspace(minimumVariance_Return, maxSharpe_Return, 100)
    for target in targetReturns:
        efficient_list.append(efficient_optimization(meanReturns, covarianceMatrix, target)['fun']) # portfolio variance

    maxSharpe_Allocation.Weights = [round(i * 100, 0) for i in maxSharpe_Allocation.Weights]
    minimumVariance_Allocation.Weights = [round(i * 100, 0) for i in minimumVariance_Allocation.Weights]

    return maxSharpe_Return, maxSharpe_STD, maxSharpe_Allocation, minimumVariance_Return, minimumVariance_STD, maxSharpe_Allocation, efficient_list, targetReturns
    
def efficient_frontier(meanReturns, covarianceMatrix):
    maxSharpe_Return, maxSharpe_STD, maxSharpe_Allocation, minimumVariance_Return, minimumVariance_STD, maxSharpe_Allocation, efficient_list, targetReturns = results(meanReturns, covarianceMatrix, risk_free_rate)

    maxSharpeRatio = go.Scatter(
        name = 'Maximum Sharpe Ratio',
        mode = 'markers',
        x = [maxSharpe_STD],
        y = [maxSharpe_Return],
        marker = dict(color = 'blue', size = 13, line = dict(width = 3, color = 'black'))
    )

    minimumVariance = go.Scatter(
        name = 'Minimum Variance',
        mode = 'markers',
        x = [minimumVariance_STD],
        y = [minimumVariance_Return],
        marker = dict(color = 'yellow', size = 13, line = dict(width = 3, color = 'black'))
    )

    EF_curve = go.Scatter(
        name = 'Efficient Frontier',
        mode = 'lines',
        x = [ef_std for ef_std in efficient_list],
        y = [target for target in targetReturns],
        line = dict(color = 'black', width = 3, dash = 'dashdot')
    )

    data = [maxSharpeRatio, minimumVariance, EF_curve]

    layout = go.Layout(
        title = 'Portfolio Optimiziation with Efficient Frontier',
        yaxis = dict(title = 'Annualized Return'),
        xaxis = dict(title = 'Annualized Standard Deviation'),
        showlegend = True,
        legend = dict(
            x = 0.75, y = 0, traceorder = 'normal', bgcolor = '#E2E2E2',
            bordercolor = 'black', borderwidth = 2),
        width = 800,
        height = 800
    )

    fig = go.Figure(data = data, layout = layout)
    return fig.show()

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

# print(minimumVariance_outputs(meanReturns, covarianceMatrix))

# print(efficient_optimization(meanReturns, covarianceMatrix, 0.2)) # minimum target return to be 20%

# print(results(meanReturns, covarianceMatrix, risk_free_rate))

efficient_frontier(meanReturns, covarianceMatrix)