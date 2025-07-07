import sys
sys.path.append(".")
import csv
import random
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime


from contents.portfolio_generator import Portfolio_Generator
from contents.simple_return import Simple_Return

class Backtest_Port():
    def __init__(self):
        #argument checks
        if len(sys.argv) == 2:
            number_check = None
            try:
                number_check = int(sys.argv[1])
            except ValueError:
                raise Exception("Please input iterations: python backtest_port.py <number of iterations>")
            if isinstance(number_check, int):
                self.arg = int(sys.argv[1])
        elif len(sys.argv) != 2:
            raise Exception('Please input only 1 argument: python backtest_port.py <number of iterations>')
        # convert csv into dictionary
        with open('inputs.csv', 'r') as file:
            reader = csv.DictReader(file)
            # Build dictionary: key = ticker, value = weight (as float)
            self.ticker_weights = {row['Tickers']: float(row['Weight']) for row in reader if row['Tickers'] and row['Weight']}
        algo_dict = {}
        # Basic Constructors
        self.today = datetime.now()
        self.universe = datetime.strptime("2000-01-01", "%Y-%m-%d")
        self.tie_in = 365
        self.interval = '1d'
        # Convert tickers with ALGO in the name into an algo dict that will be inserted into backtest method
        self.algo_dict = algo_dict
        for ticker in self.ticker_weights:
            if 'algo' in ticker.lower():
                print("Downloading ALGO dfs...")
                # Remove ' ALGO' from ticker for yfinance download if needed
                yf_ticker = ticker.replace(' ALGO', '').replace('algo', '').strip()
                self.algo_dict[ticker] = yf.download(
                    yf_ticker,
                    start=self.universe,
                    end=str(date.today() - timedelta(1)),
                    interval=self.interval,
                    multi_level_index=False,
                    ignore_tz=True
                )
        #Dict of all tickers
        self.dictionary_of_tickers = self.ticker_weights.copy()
        # Dict of tickers without algo in name
        self.ticker_list = [k for k in self.ticker_weights if 'algo' not in str(k).lower()]
        self.ticker_list = [k for k in self.dictionary_of_tickers if 'algo' not in k.lower()]
        # Download for all tickers without algo
        self.df = yf.download(self.ticker_list, start = self.universe, end=str(date.today() - timedelta(1)), interval=self.interval, multi_level_index=True, ignore_tz=True)
        self.df.dropna(inplace=True)
        # true starting dates are the dates where the latest security was IPO'd i.e not NAs
        # this is also the random number range
        self.true_start_date = self.df.index[0]
        self.true_end_date = self.df.index[-1] - timedelta(days=self.tie_in)
        print(f"Downloading SPY...")
        # You only need to download the true range for spy
        self.spydf = yf.download('SPY', start = self.true_start_date, end = self.true_end_date, interval = self.interval, multi_level_index=False, ignore_tz=True)
    def backtest(self):
        for i in range(self.arg):
            print(f"Backtest {i + 1} of {self.arg}...")
            #random dates using the true range
            random_input = random.randint(0, (self.true_end_date - self.true_start_date).days)
            input_start_date = pd.to_datetime(self.true_start_date + timedelta(days=random_input))
            input_end_date = pd.to_datetime(input_start_date + timedelta(days=self.tie_in))

            if input_end_date < self.today:
                try:
                    model = Portfolio_Generator(dictionary_of_tickers=self.dictionary_of_tickers, start=input_start_date, end=input_end_date, interval=self.interval, optional_df=self.df)
                except Exception as e:
                    print(f"Iteration {i+1}: {e}")
                    print("Skipping this iteration due to insufficient data.")
                    continue  # Skip to the next iteration
                real_start_date = model.df.index[0]  # Get the first date in the DataFrame
                real_end_date = model.df.index[-1]  # Get the last date in the DataFrame
                backtest_result = model.backtest(return_table=False, print_statement=True, model_return=True, algo_dictionary=self.algo_dict)
                backtest_sharpe = model.sharpe_ratio(return_model=True)
                #Spy modules
                spy_model = Simple_Return(ticker='SPY', interval=self.interval, start=input_start_date, end=real_end_date, optional_df=self.spydf)
                spy_result = spy_model.get_return()
                spy_sharpe = spy_model.get_sharpe()
                spy_delta = backtest_result - spy_result
                print(f"SPY Buy/Hold Result: {spy_result}%")
                #export to csv function
                def export_to_csv(backtest_result, filename=f"portfolio_backtest_results.csv"):
                    #Check for Overload Error
                    if np.isnan(backtest_sharpe):
                        print(f"Error: Errors found in backtest due to overload. Backtest #{i + 1} scrapped.")
                        return
                    else:
                        with open(filename, 'a', newline='') as csvfile:
                            writer = csv.writer(csvfile)
                            if csvfile.tell() == 0:  # Check if file is empty
                                # Write header only if the file is empty
                                writer.writerow(['Input Start Date', 'Input End Date', 'Start Date', 'End Date', 'Model Result', 'Model Sharpe', 'SPY Buy/Hold Result', 'SPY Sharpe', 'SPY Delta'])
                            writer.writerow([input_start_date, input_end_date, real_start_date, real_end_date, backtest_result, backtest_sharpe, spy_result, spy_sharpe, round(spy_delta,2)]) # data
                print("Done")
                export_to_csv(backtest_result)
            elif input_end_date >= self.today:
                print(f"End Date is not valid, no entry recorded")
                print(f"{input_end_date}")
        print("-" * 50)
        print("-" * 50)
        print("Backtest completed")
        print("-" * 50)
        print("-" * 50)
    def return_data(self):
        return self.df
    
if __name__ == "__main__":
    test = Backtest_Port()
    test.backtest()