from datetime import date, timedelta
from scipy import stats
import pandas as pd
import numpy as np
import yfinance as yf

class Simple_Return():
    def __init__(self, ticker, start = str(date.today() - timedelta(59)), end = str(date.today() - timedelta(1)), interval = "1d", optional_df = None):
# Basic Constructors
#Optinal DF is the exact same format as the below yfinance download, this functionality is used so downloaded data is not required every iteration of an algo call
        if optional_df is not None:
            self.df = optional_df
        else:
            df = yf.download(ticker, start, end, interval = interval, multi_level_index=False)
            self.df = df
        self.interval = interval
        self.start = start
        self.end = end
        self.df = self.df[self.df.index >= self.start]
        self.df = self.df[self.df.index <= self.end]
        day_count = np.arange(1, len(self.df) + 1)
        self.df['Day Count'] = day_count
        self.df['Return'] = np.log(self.df['Close']).diff() * 100
        self.df['Cumulative Return %'] = (np.exp(self.df['Return'] / 100).cumprod() - 1) * 100
        self.df.dropna(inplace=True)
        
    def get_return(self):
        return round(((self.df['Close'].iloc[-1] - self.df['Close'].iloc[0])/self.df['Close'].iloc[0]) * 100, 2)
    
    def return_df(self):
        return self.df
    
    def get_sharpe(self):
        if self.interval == "1d":
            annualized_factor = 252
        elif self.interval == "1wk":
            annualized_factor = 52
        elif self.interval == "1mo":
            annualized_factor = 12
        else:
            raise ValueError("Unsupported interval for Sharpe Ratio calculation. Use '1d', '1wk', or '1mo'.")
        buy_hold_descriptives = stats.describe(self.df['Close'].pct_change().dropna())
        buy_hold_mean = buy_hold_descriptives.mean
        buy_hold_std = buy_hold_descriptives.variance ** 0.5
        buy_hold_sharpe = buy_hold_mean / buy_hold_std * (annualized_factor ** 0.5)
        return round(buy_hold_sharpe, 6)