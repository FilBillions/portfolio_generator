import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from scipy import stats

np.set_printoptions(legacy='1.25')

class Mean_Rev_BackTest():
    def __init__(self,  
                 ticker,
                 ma1 = 9,
                 start = str(date.today() - timedelta(59)), 
                 end = str(date.today() - timedelta(1)), 
                 interval = "1d", 
                 p1 = 5, 
                 p2 = 10, 
                 mean = 50, 
                 p3 = 90, 
                 p4 = 95,
                 optional_df = None):
        
        if optional_df is not None:
            self.df = optional_df
        else:
            df = yf.download(ticker, start, end, interval = interval, multi_level_index=False, ignore_tz=True)
            self.df = df
        self.ticker = ticker
        self.interval = interval
        self.ma1 = ma1
        day_count = np.arange(1, len(self.df) + 1)
        self.df['Day Count'] = day_count
        self.df['Return %'] = (np.log(self.df['Close']).diff()) * 100
        self.df[f'{self.ma1}-day SMA'] = self.df['Close'].rolling(int(self.ma1)).mean().shift()
        self.df['Previous Close'] = self.df['Close'].shift(1)

        # --- Ratio of the close price over the moving average ---
        self.df['Ratio'] = self.df['Close'] / self.df[f'{self.ma1}-day SMA']
        self.df['Ratio'] = self.df['Ratio'].fillna(0)

        # --- intialising percentiles and areas where we'll long and short ---
        percentiles = [p1, p2, mean, p3, p4]

        # percentile calc
        filtered_ratio = self.df['Ratio'][(self.df['Ratio'] > 0) & (self.df['Ratio'].notna())]
        self.p = np.percentile(filtered_ratio, percentiles)

        self.short = self.p[4]
        self.short2 = self.p[3]
        self.long = self.p[0]
        self.long2 = self.p[1]
        self.exit = self.p[2]

    def run_algo(self, start_date=date.today().year- 1, end_date=date.today(), return_table=False):
        if self.df.index.tz is not None:
            if pd.Timestamp(start_date).tzinfo is None:
                start_date = pd.Timestamp(start_date).tz_localize('UTC')
            else:
                start_date = pd.Timestamp(start_date).tz_convert('UTC')
            if pd.Timestamp(end_date).tzinfo is None:
                end_date = pd.Timestamp(end_date).tz_localize('UTC')
            else:
                end_date = pd.Timestamp(end_date).tz_convert('UTC')
        else:
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date)

        if isinstance(start_date, int):
            post_data = self.df[self.df.index.year >= start_date]
            data_cutoff = []
        else:
            post_data = self.df[self.df.index >= start_date]
            if end_date == date.today():
                end_date = str(date.today())
            data_cutoff = self.df[self.df.index >= end_date]

        # Prepare lists to collect actions
        actions = []
        dates = []
        post_idx = self.df.index.get_loc(post_data.index[0])

        prev_action = 'No Action'  # Initialize previous action
        next_action = None  # This will hold the action for the next day
        action = 'No Action'  # <-- Ensure action is always initialized
        while post_idx < (len(self.df) - len(data_cutoff)):
            #update the conditional probability table every number of steps
            start_date = self.df.index[post_idx]
        # generate a buy signal when the ratio CROSSES below the long theshold
            if self.df['Ratio'].iloc[post_idx - 1] >= self.long and self.df['Ratio'].iloc[post_idx] < self.long:
                if prev_action != 'Sell to Open' and prev_action != 'Hold (Short)':
                    action = 'Buy to Open'
        # generate a buy signal when the ratio CROSSES below the long2 theshold
            elif self.df['Ratio'].iloc[post_idx - 1] >= self.long2 and self.df['Ratio'].iloc[post_idx] < self.long2:
                if prev_action != 'Sell to Open' and prev_action != 'Hold (Short)':
                    action = 'Buy to Open'
        # generate a sell signal when the ratio CROSSES above the short theshold
            elif self.df['Ratio'].iloc[post_idx - 1] <= self.short and self.df['Ratio'].iloc[post_idx] > self.short:
                if prev_action != 'Buy to Open' and prev_action != 'Hold (Long)':
                    action = 'Sell to Open'
        # generate a sell signal when the ratio CROSSES above the short2 theshold
            elif self.df['Ratio'].iloc[post_idx - 1] <= self.short2 and self.df['Ratio'].iloc[post_idx] > self.short2:
                if prev_action != 'Buy to Open' and prev_action != 'Hold (Long)':
                    action = 'Sell to Open'
        # generate a Sell to Close signal when the ratio crosses above the exit theshold and we are in a long position or holding a long position
            elif (prev_action == 'Buy to Open' or prev_action == 'Hold (Long)') and self.df['Ratio'].iloc[post_idx - 1] < self.exit and self.df['Ratio'].iloc[post_idx] >= self.exit:
                action = 'Sell to Close'
        # generate a Buy to Close signal when the ratio crosses below the exit theshold and we are in a short position or holding a short position
            elif (prev_action == 'Sell to Open' or prev_action == 'Hold (Short)') and self.df['Ratio'].iloc[post_idx - 1] > self.exit and self.df['Ratio'].iloc[post_idx] <= self.exit:
                action = 'Buy to Close'
        # Hold long if we are in a long position and we have not yet hit the exit threshold
            elif (prev_action == 'Buy to Open' or prev_action == 'Hold (Long)') and self.df['Ratio'].iloc[post_idx] < self.exit:
                action = 'Hold (Long)'
        # Hold short if we are in a short position and we have not yet hit the exit threshold
            elif (prev_action == 'Sell to Open' or prev_action == 'Hold (Short)') and self.df['Ratio'].iloc[post_idx] > self.exit:
                action = 'Hold (Short)'
            else:
                if prev_action == "Buy to Open" or prev_action == "Hold (Long)":
                    action = 'Sell to Close'
                elif prev_action == "Sell to Open" or prev_action == "Hold (Short)":
                    action = 'Buy to Close'
                elif prev_action == "No Action" or prev_action == "Buy to Close" or prev_action == "Sell to Close":
                    action = 'No Action'
                else:
                    action = 'Error'

        # Only append the action for the previous day (to avoid lookahead bias)
            if next_action is not None:
                dates.append(start_date)
                actions.append(next_action)

            prev_action = action
            next_action = action
            post_idx += 1
        
        df_actions = pd.DataFrame({'Date': dates, 'Action': actions})
        self.df = self.df.join(df_actions.set_index('Date'), how='left')
        self.df['Buy Signal'] = np.where(self.df['Action'] == 'Buy to Open', self.df['Close'].shift(1), (np.where(self.df['Action'] == 'Buy to Close', self.df['Close'].shift(1), np.nan)))
        # Previous close at Sell signals
        self.df['Sell Signal'] = np.where(self.df['Action'] == 'Sell to Open', self.df['Close'].shift(1), (np.where(self.df['Action'] == 'Sell to Close', self.df['Close'].shift(1), np.nan)))

       # remove rows with NaN in action
        self.df.dropna(subset=['Action'], inplace=True)

        if return_table:
            print(f"Total Trades: {df_actions['Action'].value_counts()['Buy to Open'] + df_actions['Action'].value_counts()['Sell to Open']}")
            print(f"Buys/Sells {df_actions['Action'].value_counts()}")
            return self.df
        
    def backtest_percent_of_equity(self, print_statement=True, return_table=False, model_return=False, buy_hold=False, return_model_df=False):
        # this backtest assumes that we are using a percentage of our equity to trade, rather than a fixed amount
        initial_investment = 10000
        cash = initial_investment
        cash_spent = 0
        total_cash_spent = 0
        shares = 0
        long_value = 0
        short_value = 0
        trade_weight = .5 # for every trade signal, we will invest 10% of our cash
        portfolio_value = []

        share_cost = self.df["Previous Close"].iloc[0]
        num_shares = initial_investment / share_cost
        self.df['Buy/Hold Value'] = num_shares * self.df['Close']
        self.df['Model Value'] = 0
    
        # Iterate through the DataFrame
        for i in range(0,len(self.df)):
            action = self.df['Action'].iloc[i]
            price = self.df['Previous Close'].iloc[i]

            if action == 'Buy to Open' and cash > 0:
                shares = ((cash * trade_weight) / price) + shares
                cash_spent = (cash * trade_weight)
                total_cash_spent += cash_spent
                avg_price = total_cash_spent / shares
                long_price = avg_price
                long_value = shares * price
                cash = cash - cash_spent
            elif action == 'Sell to Close' and shares > 0:
                cash = (shares * long_price) + ((shares * long_price) * ((price - long_price) / long_price)) + cash
                shares = 0
                avg_price = 0
                long_price = 0
                long_value = 0
                cash_spent = 0
                total_cash_spent = 0
            elif action == 'Sell to Open' and cash > 0:
                shares = ((cash * trade_weight) / price) + shares
                cash_spent = (cash * trade_weight)
                total_cash_spent += cash_spent
                avg_price = total_cash_spent / shares
                short_price = avg_price
                short_value = shares * price
                cash = cash - cash_spent
            elif action == 'Buy to Close' and shares > 0:
                cash = (shares * short_price) - ((shares * short_price) * ((price - short_price) / short_price)) + cash
                shares = 0
                avg_price = 0
                short_value = 0
                short_price = 0
                cash_spent = 0
                total_cash_spent = 0
            elif action == "Hold (Short)":
                short_value = shares * price
            elif action == 'Hold (Long)':
                long_value = shares * price
    
            model_value = (cash + long_value + short_value)
            portfolio_value.append(model_value)
        self.df['Model Value'] = portfolio_value
                #dropping unnecessary columns
        if 'Volume' in self.df.columns:
                self.df.drop(columns=['Volume'], inplace = True)
        if 'Previous Bin' in self.df.columns:
                self.df.drop(columns=['Previous Bin'], inplace = True)
        if 'Current Bin' in self.df.columns:
                self.df.drop(columns=['Current Bin'], inplace = True)
        
        if print_statement:
            print(f"{self.ticker} Buy/Hold Result: {round(((self.df['Buy/Hold Value'].iloc[-1] - self.df['Buy/Hold Value'].iloc[0])/self.df['Buy/Hold Value'].iloc[0]) * 100, 2)}%")
            print(f"{self.ticker} Model Result: {round(((self.df['Model Value'].iloc[-1] - self.df['Model Value'].iloc[0])/self.df['Model Value'].iloc[0]) * 100, 2)}%")
            print(f" from {self.df.index[0]} to {self.df.index[-1]}")
        if return_table:
            return self.df
        if model_return:
            return round(((self.df['Model Value'].iloc[-1] - self.df['Model Value'].iloc[0])/self.df['Model Value'].iloc[0]) * 100, 2)
        if buy_hold:
            return round(((self.df['Buy/Hold Value'].iloc[-1] - self.df['Buy/Hold Value'].iloc[0])/self.df['Buy/Hold Value'].iloc[0]) * 100, 2)
        if return_model_df:
            return self.df['Model Value']

    def backtest_cash(self, print_statement=True, return_table=False, model_return=False, buy_hold=False, return_model_df=False):
        # this backtest assumes we are using 10% of our cash to trade at any given time
        # we have a max pyramiding of 10 positions at a time
        initial_investment = 10000
        cash = initial_investment
        trade_amt = 0
        beginning_cash = 0
        total_cash_spent = 0
        shares = 0
        long_value = 0
        short_value = 0
        portfolio_value = []

        share_cost = self.df["Previous Close"].iloc[0]
        num_shares = initial_investment / share_cost
        self.df['Buy/Hold Value'] = num_shares * self.df['Close']
        self.df['Model Value'] = 0
    
        # Iterate through the DataFrame
        for i in range(0,len(self.df)):
            action = self.df['Action'].iloc[i]
            price = self.df['Previous Close'].iloc[i]

            if action == 'Buy to Open' and cash > 0:
                beginning_cash = cash + total_cash_spent
                trade_amt = beginning_cash * 0.1
                shares = (trade_amt / price) + shares
                total_cash_spent += trade_amt
                avg_price = total_cash_spent / shares
                long_value = shares * price
                cash = cash - trade_amt
            elif action == 'Sell to Close' and shares > 0:
                cash = (shares * avg_price) + ((shares * avg_price) * ((price - avg_price) / avg_price)) + cash
                beginning_cash = 0
                trade_amt = 0
                shares = 0
                total_cash_spent = 0
                avg_price = 0
                long_value = 0
            elif action == 'Sell to Open' and cash > 0:
                beginning_cash = cash + total_cash_spent
                trade_amt = beginning_cash * 0.1
                shares = (trade_amt / price) + shares
                total_cash_spent += trade_amt
                avg_price = total_cash_spent / shares
                short_value = shares * price
                cash = cash - trade_amt
            elif action == 'Buy to Close' and shares > 0:
                cash = (shares * avg_price) - ((shares * avg_price) * ((price - avg_price) / avg_price)) + cash
                beginning_cash = 0
                trade_amt = 0
                shares = 0
                total_cash_spent = 0
                avg_price = 0
                short_value = 0
            elif action == "Hold (Short)":
                short_value = shares * price
            elif action == 'Hold (Long)':
                long_value = shares * price
    
            model_value = (cash + long_value + short_value)
            portfolio_value.append(model_value)
        self.df['Model Value'] = portfolio_value
                #dropping unnecessary columns
        if 'Volume' in self.df.columns:
                self.df.drop(columns=['Volume'], inplace = True)
        if 'Previous Bin' in self.df.columns:
                self.df.drop(columns=['Previous Bin'], inplace = True)
        if 'Current Bin' in self.df.columns:
                self.df.drop(columns=['Current Bin'], inplace = True)
        
        if print_statement:
            print(f"{self.ticker} Buy/Hold Result: {round(((self.df['Buy/Hold Value'].iloc[-1] - self.df['Buy/Hold Value'].iloc[0])/self.df['Buy/Hold Value'].iloc[0]) * 100, 2)}%")
            print(f"{self.ticker} Model Result: {round(((self.df['Model Value'].iloc[-1] - self.df['Model Value'].iloc[0])/self.df['Model Value'].iloc[0]) * 100, 2)}%")
            print(f" from {self.df.index[0]} to {self.df.index[-1]}")
        if return_table:
            # see all rows
            return self.df
        if model_return:
            return round(((self.df['Model Value'].iloc[-1] - self.df['Model Value'].iloc[0])/self.df['Model Value'].iloc[0]) * 100, 2)
        if buy_hold:
            return round(((self.df['Buy/Hold Value'].iloc[-1] - self.df['Buy/Hold Value'].iloc[0])/self.df['Buy/Hold Value'].iloc[0]) * 100, 2)
        if return_model_df:
            return self.df['Model Value'] 
              
    def sharpe_ratio(self, return_model=True, return_buy_hold=False):
        # factor answers the question: how many of this interval are in the total timespan
        if self.interval == "1d":
            #There are 252 trading days in a year
            annualized_factor = 252
        elif self.interval == "1wk":
            #52 weeks in a year
            annualized_factor = 52
        elif self.interval == "1mo":
            #12 months in a year
            annualized_factor = 12
        else:
            annualized_factor = 1
        model_descriptives = stats.describe(self.df['Model Value'].pct_change().dropna())
        model_mean = model_descriptives.mean
        model_std = model_descriptives.variance ** 0.5
        model_sharpe = model_mean / model_std * (annualized_factor ** 0.5)
        buy_hold_descriptives = stats.describe(self.df['Buy/Hold Value'].pct_change().dropna())
        buy_hold_mean = buy_hold_descriptives.mean
        buy_hold_std = buy_hold_descriptives.variance ** 0.5
        buy_hold_sharpe = buy_hold_mean / buy_hold_std * (annualized_factor ** 0.5)
        if return_buy_hold:
            return round(buy_hold_sharpe, 6)
        if return_model:
            return round(model_sharpe, 6)