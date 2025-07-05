import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from scipy import stats
from scipy.stats import norm
import sys
import os
sys.path.append(os.path.dirname(__file__))
from charts import normal
from get_row_label import get_row_label

# change based on what conditional probability function you want to use
# function should always be nameed conditional_probability
from current_given_previous_func import conditional_probability

np.set_printoptions(legacy='1.25')

# Long and Short added

# Creates a Conditional Probability Object
# This object contains the stock data, descriptive statistics, and methods for calculating conditional probabilities.
# The algorithm and backtest methods are generic and support any conditional probability function

class Conditional_Probability():
    def __init__(self, ticker, start = str(date.today() - timedelta(59)), end = str(date.today() - timedelta(1)), interval = "1d", optional_df = None):
    # Basic Constructors
    #Optinal DF is the exact same format as the below yfinance download, this functionality is used so downloaded data is not required every iteration of an algo call
        if optional_df is not None:
            self.df = optional_df
        else:
            df = yf.download(ticker, start, end, interval = interval, multi_level_index=False)
            self.df = df
        day_count = np.arange(1, len(self.df) + 1)
        self.df['Day Count'] = day_count
        self.ticker = ticker
        self.close = self.df['Close']
        self.percent_change = np.log(self.close).diff() * 100
        self.percent_change.dropna(inplace = True)
        self.df['Return'] = self.percent_change
        self.interval = interval
    
    # - - - Descriptives - - - 
        n , minmax, mean, var, skew, kurt = stats.describe(self.percent_change)
        mini, maxi = minmax
        std = var ** .5
        self.random_sample = norm.rvs(mean, std, n)
        self.n = n
        self.mean = mean
        self.var = var
        self.skew = skew
        self.kurt = kurt
        self.mini = mini
        self.maxi = maxi
        self.std = std

    # - - - NORMAL CALCS - - -
        # overlay is your X value
        self.overlay = np.linspace(self.mini, self.maxi, 100)
        # p is simply your p value for normal calcs
        self.p = norm.pdf(self.overlay,self.mean,self.std)

# Calculate Previous Close, used only for avoiding lookahead bias.
# This Previous Close is not used in calculating any conditional probabilities.
        self.df['Previous Close'] = self.df['Close'].shift(1)

        self.df.dropna(inplace=True)

    def normal(self):
    # descriptive statistics
        print(stats.describe(self.percent_change))
        random_test = stats.kurtosistest(self.random_sample)
        stock_test = stats.kurtosistest(self.percent_change)
        print('Null: The Sample is Normally Distributed')
        print('If P-Value < .05: Reject H0; If P-Value >= .05: Cannot Reject H0')
        print(f'{"-"*60}')
        print(f"Random Test: Statistic: {round(random_test[0], 2)}, P-Value: {round(random_test[1], 2)}")
        if random_test[1] >= .05:
            print('We cannot reject H0')
        else:
            print('We can reject H0')
        print(f'{"-"*60}')
        print(f"{self.ticker} Test: Statistic: {round(stock_test[0], 2)}, P-Value: {round(stock_test[1], 2)}")
        if stock_test[1] >= .05:
            print('We cannot reject H0, this is probably Normally Distributed')
        else:
            print('We can reject H0, this is probably not Normally Distributed')
        return normal(self)

    def probability(self, threshold):
        # simple normal distribution probability calculation
        if threshold == None:
            raise ValueError("No Threshold")
        if threshold <= 0:
            probability = 1 - (norm.sf(threshold, loc=self.mean, scale=self.std))
            print(f"Probability of {self.ticker} losing {threshold}% in {self.interval} is {round(probability*100,2):.2f}%")
        else:
            probability = norm.sf(threshold, loc=self.mean, scale=self.std)
            print(f"Probability of {self.ticker} gaining {threshold}% in {self.interval} is {round(probability*100,2):.2f}%")
    
    def run_algo(self, target_probability=.55, start_date=date.today().year- 1, end_date=date.today(), step_input=5, return_table=False):
        # - - - Run the Algorithm - - -
        # - - - Initialize post data and pre data sets - - -
        # - - - We only use data from before the specified start date - - -
        # Ensure start_date and end_date are timezone-aware and in UTC
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
        probs = []

        # Use integer indices instead of slicing DataFrames
        post_idx = self.df.index.get_loc(post_data.index[0])

        prev_action = None
        next_action = None  # This will hold the action for the next day
        action = 'No Action'  # <-- Ensure action is always initialized
        conditional_prob_prev = None  # Also initialize this
        # This is the algorithm loop
        while post_idx < (len(self.df) - len(data_cutoff)):
            if (post_idx - self.df.index.get_loc(post_data.index[0])) % step_input == 0: # every x steps, recalculate the conditional probability table using new data
                current_pre_data = self.df.iloc[:post_idx]                      # this helps with runtime
                algo_df = conditional_probability(current_pre_data,print_statement=False)
            start_date = self.df.index[post_idx]
            last_return = self.df['Return'].iloc[post_idx - 1]
            row_label = get_row_label(last_return)
            conditional_prob_long = algo_df.loc[row_label, 'Positive']
            conditional_prob_short = algo_df.loc[row_label, 'Negative']
            # Decide action for the NEXT day
            if conditional_prob_long > conditional_prob_short:
                if conditional_prob_long > target_probability:
                    #Condition to Long or Stay Long
                    if prev_action == "Hold (Long)" or prev_action == "Buy to Open":
                        action = "Hold (Long)"
                    elif prev_action == "No Action" or prev_action == "Sell to Close" or prev_action == "Buy to Close":
                        action = "Buy to Open"
                    # Rare Scenarios where probabilites are stacks and short trades need to be kept open or closed
                    elif conditional_prob_short > target_probability:
                        if prev_action == "Hold (Short)" or prev_action == "Sell to Open":
                            action = "Hold (Short)"
                    elif conditional_prob_short < target_probability:
                        if prev_action == "Hold (Short)" or prev_action == "Sell to Open":
                            action = "Buy to Close"
                elif conditional_prob_long < target_probability:
                    #If target prob is greater, close any trade possible
                    if prev_action == "Hold (Long)" or prev_action == "Buy to Open":
                        action = "Sell to Close"    
                    elif prev_action == "Sell to Open" or prev_action == "Hold (Short)":
                        action = "Buy to Close"
                    elif prev_action == "Buy to Close" or prev_action == "Sell to Close":
                        action = "No Action"
            elif conditional_prob_long < conditional_prob_short:
                if conditional_prob_short > target_probability:
                    #Conditions to Short or Hold Short
                    if prev_action == "Hold (Short)" or prev_action == "Sell to Open":
                        action = "Hold (Short)"
                    elif prev_action == "No Action" or prev_action == "Buy to Close" or prev_action == "Sell to Close":
                        action = "Sell to Open"
                    # Rare Scenarios where probabilites are stacks and long trades need to be kept open or closed
                    elif conditional_prob_long > target_probability:
                        if prev_action == "Hold (Long)" or prev_action == "Buy to Open":
                            action = "Hold (Long)"
                    elif conditional_prob_long < target_probability:
                        if prev_action == "Hold (Long)" or prev_action == "Buy to Open":
                            action = "Sell to Close"
                elif conditional_prob_short < target_probability:
                    #If target prob is greater, close any trade possible
                    if prev_action == "Hold (Short)" or prev_action == "Sell to Open":
                        action = "Buy to Close" 
                    elif prev_action == "Buy to Open" or prev_action == "Hold (Long)":
                        action = "Sell to Close"
                    elif prev_action == "Sell to Close" or prev_action == "Buy to Close":
                        action = "No Action"
            else:
                # Close any trade if none of the above conditions have been fulfilled
                if prev_action == "Sell to Open" or prev_action == "Hold (Short)":
                    action = "Buy to Close"
                elif prev_action == "Buy to Open" or prev_action == "Hold (Long)":
                    action = "Sell to Close"
                # If none ofthe above conditions are met, all trades should stay closed
                elif prev_action == "Buy to Close" or prev_action == "Sell to Close" or prev_action == "No Action":
                    action = "No Action"
                else:
                    # If this pops up on the tape, it means the situation is not accounted for
                    action = "Error"

            # Only append the action for the previous day (to avoid lookahead bias)
            if next_action is not None:
                dates.append(start_date)
                actions.append(next_action)
                probs.append(conditional_prob_prev)  # Save the previous day's probability

            # Prepare for next iteration
            prev_action = action
            next_action = action
            #put an if statement here
            if conditional_prob_long > conditional_prob_short:
                conditional_prob_prev = conditional_prob_long
            if conditional_prob_short > conditional_prob_long:
                conditional_prob_prev = conditional_prob_short

            post_idx += 1

        # Create the actions DataFrame once at the end
        df_actions = pd.DataFrame({'Date': dates, 'Action': actions, 'Probability > 0%': probs})

        self.df = self.df.join(df_actions.set_index('Date'), how='left')
        self.df['Probability > 0%'] = self.df['Probability > 0%'].ffill()
        # Previous close at Buy signals
        self.df['Buy Signal'] = np.where(self.df['Action'] == 'Buy to Open', self.df['Close'].shift(1), (np.where(self.df['Action'] == 'Buy to Close', self.df['Close'].shift(1), np.nan)))
        # Previous close at Sell signals
        self.df['Sell Signal'] = np.where(self.df['Action'] == 'Sell to Open', self.df['Close'].shift(1), (np.where(self.df['Action'] == 'Sell to Close', self.df['Close'].shift(1), np.nan)))

       # remove rows with NaN in action
        self.df.dropna(subset=['Action'], inplace=True)

        if return_table:
            print(f"Buys/Sells {df_actions['Action'].value_counts()}")
            return round(self.df,4)
    
    def backtest(self, print_statement=True, return_table=False, model_return=False, buy_hold=False, return_model_df=False):
        initial_investment = 10000
        cash = initial_investment
        shares = 0
        long_value = 0
        short_value = 0
        portfolio_value = []

        # Calculate Buy/Hold Value
        # previous Close is used to avoid lookahead bias
        share_cost = self.df['Previous Close'].iloc[0]
        num_shares = initial_investment / share_cost
        self.df['Buy/Hold Value'] = num_shares * self.df['Close']
        self.df['Model Value'] = 0
        # Iterate through the DataFrame
        for i in range(0,len(self.df)):
            action = self.df['Action'].iloc[i]
            price = self.df['Previous Close'].iloc[i]

            if action == 'Buy to Open' and cash > 0:
                shares = cash/price
                long_price = price
                long_value = shares * price
                cash = 0
            elif action == 'Sell to Close' and shares > 0:
                cash = (shares * long_price) + ((shares * long_price) * ((price - long_price) / long_price))
                shares = 0
                long_price = 0
                long_value = 0
            elif action == 'Sell to Open' and cash > 0:
                short_price = price
                shares = cash/price
                short_value = shares * price
                cash = 0
            elif action == 'Buy to Close' and shares > 0:
                cash = (shares * short_price) - ((shares * short_price) * ((price - short_price) / short_price))
                shares = 0
                short_value = 0
                short_price = 0
            elif action == "Hold (Short)":
                short_value = (shares * short_price) - ((shares * short_price) * ((price - short_price) / short_price))
            elif action == 'Hold (Long)':
                long_value = (shares * long_price) + ((shares * long_price) * ((price - long_price) / long_price))
    
            model_value = (cash + short_value + long_value)
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
