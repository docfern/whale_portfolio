 #  A Whale off the Port(folio)

import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
%matplotlib inline

#Data Cleaning

## Whale Returns

#Reading whale returns
whale_returns_csv = Path("../Resources/whale_returns.csv")
whale_returns = pd.read_csv(whale_returns_csv, index_col="Date", infer_datetime_format=True, parse_dates=True)
whale_returns.head()

#Count nulls
whale_returns.isnull().sum()

#Drop nulls
whale_returns.dropna(inplace=True)
whale_returns.isnull().sum()

## Algorithmic Daily Returns

#Reading algorithmic returns
algo_returns_csv = Path("../Resources/algo_returns.csv")
algo_returns = pd.read_csv(algo_returns_csv, index_col="Date", infer_datetime_format=True, parse_dates=True)
algo_returns.head()

#Count nulls
algo_returns.isnull().sum()

#Drop nulls
algo_returns.dropna(inplace=True)
algo_returns.isnull().sum()

## S&P 500 Returns

#Reading S&P 500 Closing Prices
sp500_history_csv = Path("../Resources/sp500_history.csv")
sp500_history = pd.read_csv(sp500_history_csv, index_col="Date", infer_datetime_format=True, parse_dates=True)
sp500_history.head()

#Check Data Types
sp500_history.dtypes

#Fix Data Types
sp500_history["Close"] = sp500_history["Close"].str.replace("$", "")
sp500_history["Close"] = sp500_history["Close"].astype("float")
sp500_history.dtypes

#Calculate Daily Returns
daily_returns_sp500 = sp500_history.pct_change()
daily_returns_sp500.head()

#Drop nulls
daily_returns_sp500.dropna(inplace=True)
daily_returns_sp500.head()

#Rename Column
daily_returns_sp500 = daily_returns_sp500.rename(columns={
    "Close": "S&P 500"})
daily_returns_sp500.head()

## Combine Whale, Algorithmic, and S&P 500 Returns

combined_df = pd.concat([whale_returns, algo_returns, daily_returns_sp500], axis="columns", join="inner")
combined_df.head()

# Portfolio Analysis

## Performance

#Plot daily returns
combined_df.plot(figsize=(20,10), title="Daily Returns")

#Plot cumulative returns
cumulative_returns = (1 + combined_df).cumprod() - 1
cumulative_returns.plot(figsize=(20,10), title="Cumulative Returns")

## Risk

#Box plot to visually show risk
combined_df.plot(kind="box", figsize=(20,10), title="Portfolio Risk")

#Daily Standard Deviations
#Calculate the standard deviation for each portfolio. Which portfolios are riskier than the S&P 500?
portfolio_risk = combined_df.std()
portfolio_risk

#Determine which portfolios are riskier than the S&P 500
SP_500_Risk = portfolio_risk[-1]
portfolio_risk.to_frame()
portfolio_risk.apply(lambda x : True if x> SP_500_Risk else False)

#Calculate the annualized standard deviation (252 trading days)
annual_portfolio_risk = combined_df.std() * np.sqrt (252)
annual_portfolio_risk

## Rolling Statistics

#Calculate and plot the rolling standard deviation for the S&PP 500 using a 21 day window
combined_df.rolling(window=21).std().plot(figsize=(20,10), title="21 Day Rolling Standard Deviation")

#Correlation
price_correlation = combined_df.corr()
price_correlation

#Calculate Beta for a single portfolio compared to the total market (S&P 500)
covariance = combined_df["BERKSHIRE HATHAWAY INC"].cov(combined_df["S&P 500"])
covariance
variance = combined_df['S&P 500'].var()
berkshire_beta = covariance / variance
rolling_covariance = combined_df['BERKSHIRE HATHAWAY INC'].rolling(window=60).cov(combined_df['S&P 500'])
rolling_variance = combined_df['S&P 500'].rolling(window=60).var()
rolling_beta = rolling_covariance / rolling_variance
rolling_beta.plot(figsize=(20, 10), title='Berkshire Hathaway Inc. Beta')

### Challenge: Exponentially Weighted Average 

combined_df.ewm(halflife=21)

## Sharpe Ratios

#Annualzied Sharpe Ratios
sharpe_ratios = (combined_df.mean() * 252) / (combined_df.std() * np.sqrt(252))
sharpe_ratios

#Visualize the sharpe ratios as a bar plot
sharpe_ratios.plot(kind="bar")

# Portfolio Returns

#Read the first stock
aapl_historical_csv = Path("../Resources/aapl_historical.csv")
aapl_historical = pd.read_csv(aapl_historical_csv, index_col="Trade DATE", infer_datetime_format=True, parse_dates=True)
aapl_historical.head()

#Read the second stock
cost_historical_csv = Path("../Resources/cost_historical.csv")
cost_historical = pd.read_csv(cost_historical_csv, index_col="Trade DATE", infer_datetime_format=True, parse_dates=True)
cost_historical.head()

#Read the third stock
goog_historical_csv = Path("../Resources/goog_historical.csv")
goog_historical = pd.read_csv(goog_historical_csv, index_col="Trade DATE", infer_datetime_format=True, parse_dates=True)
goog_historical.head()

#Concatenate all stocks into a single DataFrame
custom_df = pd.concat([aapl_historical, cost_historical, goog_historical], axis="rows", join="outer")
custom_df.head()

#Reset the index
custom_df.reset_index()

#Pivot the Data so that the stock tickers are the columns, the dates are the index, and the 
#values are the closing prices
custom_df = custom_df.pivot(columns="Symbol", values="NOCP")
custom_df.head()

#Drop Nulls
custom_daily_returns = custom_df.pct_change()
custom_daily_returns.dropna(inplace=True)
custom_daily_returns.head()

## Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock

#Calculate weighted portfolio returns
weights = [1/3, 1/3, 1/3]
custom_portfolio_returns = custom_daily_returns.dot(weights)
custom_portfolio_returns.head()

## Join your portfolio returns to the DataFrame that contains all of the portfolio returns

all_portfolio_returns = pd.concat([combined_df, custom_portfolio_returns], axis="columns", join="inner")
x = all_portfolio_returns.columns[-1]
all_portfolio_returns = all_portfolio_returns.rename(columns={x:"Custom"})
all_portfolio_returns.columns

#Only compare dates where the new, custom portfolio has dates
all_portfolio_returns.isnull()
all_portfolio_returns.dropna(axis=0, how="any", inplace=True)
all_portfolio_returns

#Risk
all_portfolio_risk = all_portfolio_returns.std() * np.sqrt (252)
all_portfolio_risk

#Rolling
all_portfolio_returns.rolling(window=21).std().plot(figsize=(20,10), title="21 Day Rolling Standard Deviation")

#Beta
covariance_all = all_portfolio_returns['Custom'].cov(all_portfolio_returns['S&P 500'])
covariance_all
variance_all = all_portfolio_returns['S&P 500'].var()
berkshire_beta = covariance / variance
rolling_covariance_all = all_portfolio_returns['Custom'].rolling(window=60).cov(all_portfolio_returns['S&P 500'])
rolling_variance_all = all_portfolio_returns['S&P 500'].rolling(window=60).var()
rolling_beta_all = rolling_covariance_all / rolling_variance_all
rolling_beta_all.plot(figsize=(20, 10), title='Custom Portfolio Beta')

#Annualzied Sharpe Ratios
sharpe_ratios_all = (all_portfolio_returns.mean() * 252) / (all_portfolio_returns.std() * np.sqrt(252))
sharpe_ratios_all

#Visualize the sharpe ratios as a bar plot
sharpe_ratios_all.plot(kind="bar")

## Include correlation analysis to determine which stocks (if any) are correlated
#Correlation
all_portfolio_returns.corr()