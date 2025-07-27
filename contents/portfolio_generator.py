from datetime import date, timedelta, datetime
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'algo'))

np.set_printoptions(legacy='1.25')

class Portfolio_Generator():
    def __init__(self, dictionary_of_tickers, start=str(date.today() - timedelta(59)), end=str(date.today() - timedelta(1)), interval='1d', optional_df=None):   
        # basic constructors
        self.start = start
        self.end = end
        self.interval = interval
        # distinquish dictionary of tickers vs dictionary with ALGOs
        self.dictionary_of_tickers = dictionary_of_tickers
        self.full_dict = self.dictionary_of_tickers
        # Remove all keys that start with 'algo' from the dictionary
        self.dictionary_of_tickers = {k: v for k, v in self.dictionary_of_tickers.items() if 'algo' not in str(k).lower()}
        self.ticker_list = list(self.dictionary_of_tickers)
        if optional_df is not None:
            self.df = optional_df
            #make sure were not processing the entire universe of data and are sticking to the targeted data range
            self.df = self.df[self.df.index >= start]
            self.df = self.df[self.df.index <= end]
        else:
            df = yf.download(self.ticker_list, start, end, interval=interval, multi_level_index=True, ignore_tz=True)
            self.df = df
        #DropNA before return calcs
        self.df.dropna(inplace=True)
        # Additions to df that dont affect the backtest
        day_count = np.arange(1, len(self.df) + 1)
        self.df['Day Count'] = day_count
        for ticker in self.ticker_list:
            if ('Close', ticker) in self.df.columns:
                self.df[('Return', ticker)] = np.log(self.df[('Close', ticker)]).diff() * 100
                self.df[('Cumulative Return', ticker)] = (np.exp(self.df['Return', ticker] / 100).cumprod() - 1) * 100
                self.df[('Previous Close', ticker)] = self.df[('Close', ticker)].shift(1)
            else:
                print(f"Warning: No Close data for {ticker}, skipping Return calculation.")
        self.df = self.df.sort_index(axis=1)
        self.df.dropna(inplace=True)
    
    def return_df(self):
        return self.df
    
    def backtest(self, print_statement=True, return_table=False, model_return=False, algo_dictionary=None):
        initial_investment = 10000
        self.port_df = pd.DataFrame()
        self.port_df['Model Value'] = 0
        sum_of_weightings = 0

        if algo_dictionary is not None:
            #declare what algo you want to import
            from mean_rev_strategy import Mean_Rev_BackTest # type: ignore
            # in the future when we have more algos, well have to do a dictionary within a dictionary.
            if isinstance(algo_dictionary, dict):
                for dict_ticker in algo_dictionary:
                    algo_df = Mean_Rev_BackTest(ticker=None, ma1=50, optional_df=algo_dictionary[dict_ticker])
                    algo_df.run_algo(start_date=self.start, end_date=self.end, return_table=False)
                    algo_value = algo_df.backtest_cash(print_statement=False, return_model_df=True)
                    self.port_df[f'{dict_ticker} Value'] = algo_value * self.full_dict[f'{dict_ticker}'] # <<< Fix input weighting
            else:
                raise Exception('algo_dictionary is not a dictionary, please use a dictionary. Format <Ticker : yfinance download information>')
        #Math for tickers
        for ticker in self.ticker_list:
            share_cost = self.df[('Previous Close', ticker)].iloc[0]
            num_shares = (initial_investment * self.dictionary_of_tickers[ticker])/ share_cost
            self.port_df[(f'{ticker} Value')] = num_shares * self.df[('Close', ticker)]
            sum_of_weightings += self.dictionary_of_tickers[ticker]
        # Check the weightings
        if sum_of_weightings > 1:
            raise Exception(f'Sum of Portfolio Weightings is not 100% --- Weight: {sum_of_weightings}')
        # Sum the columns
        value_cols = [col for col in self.port_df.columns if col.endswith('Value') and col != 'Model Value']
        self.port_df['Model Value'] = self.port_df[value_cols].sum(axis=1)

        if print_statement:
            print(f"Model Portfolio Result: {round(((self.port_df['Model Value'].iloc[-1] - self.port_df['Model Value'].iloc[0])/self.port_df['Model Value'].iloc[0]) * 100, 2)}%")
            print(f" from {self.port_df.index[0]} to {self.port_df.index[-1]}")
            for col in self.port_df.columns:
                if col != 'Model Value':
                    print(f"{col} return: {round(((self.port_df[col].iloc[-1] - self.port_df[col].iloc[0])/self.port_df[col].iloc[0]) * 100, 2)}%")
        if return_table:
            return self.port_df
        if model_return:
            return round(((self.port_df['Model Value'].iloc[-1] - self.port_df['Model Value'].iloc[0])/self.port_df['Model Value'].iloc[0]) * 100, 2)
    
    def sharpe_ratio(self, return_model=True):
        annualized_factor = 252
        model_descriptives = stats.describe(self.port_df['Model Value'].pct_change().dropna())
        model_mean = model_descriptives.mean
        model_std = model_descriptives.variance ** 0.5
        model_sharpe = model_mean / model_std * (annualized_factor ** 0.5)
        if return_model:
            return round(model_sharpe, 6)