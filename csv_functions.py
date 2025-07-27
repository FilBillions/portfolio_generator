import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats

np.set_printoptions(legacy='1.25')

def return_hist(input, backtest=True, buy_hold=False, spy=False, csv_off=False):
    fig1 = plt.figure(figsize=(12, 6))
    if csv_off is False:
        df = pd.read_csv(input)
    if csv_off is True:
        df = input
    if df.empty:
        print("CSV file is empty. Please run the backtest first.")
        return

    #variables
    backtest_mini, backtest_maxi = df['Model Result'].min(), df['Model Result'].max()
    backtest_mean = df['Model Result'].mean()
    backtest_std = df['Model Result'].std()
    backtest_overlay = np.linspace(backtest_mini, backtest_maxi, 100)
    backtest_p = norm.pdf(backtest_overlay, backtest_mean, backtest_std)
    if 'Buy/Hold Result' in df.columns:
        buy_hold_mini, buy_hold_maxi = df['Buy/Hold Result'].min(), df['Buy/Hold Result'].max()
        buy_hold_mean = df['Buy/Hold Result'].mean()
        buy_hold_std = df['Buy/Hold Result'].std()
        buy_hold_overlay = np.linspace(buy_hold_mini, buy_hold_maxi, 100)
        buy_hold_p = norm.pdf(buy_hold_overlay, buy_hold_mean, buy_hold_std)
    if 'SPY Buy/Hold Result' in df.columns:
        spy_buy_hold_mini, spy_buy_hold_maxi = df['SPY Buy/Hold Result'].min(), df['SPY Buy/Hold Result'].max()
        spy_buy_hold_mean = df['SPY Buy/Hold Result'].mean()
        spy_buy_hold_std = df['SPY Buy/Hold Result'].std()
        spy_buy_hold_overlay = np.linspace(spy_buy_hold_mini, spy_buy_hold_maxi, 100)
        spy_buy_hold_p = norm.pdf(spy_buy_hold_overlay, spy_buy_hold_mean, spy_buy_hold_std)
    print("-" * 50)
    print(f"n = {len(df)}")
    print("-" * 50)
    print(f'90% Confidence Interval: {tuple(round(x, 4) for x in stats.norm.interval(.90, loc=backtest_mean, scale=(backtest_std/(len(df) ** .5))))}')
    print(f'95% Confidence Interval: {tuple(round(x, 4) for x in stats.norm.interval(.95, loc=backtest_mean, scale=(backtest_std/(len(df) ** .5))))}')
    print(f'98% Confidence Interval: {tuple(round(x, 4) for x in stats.norm.interval(.98, loc=backtest_mean, scale=(backtest_std/(len(df) ** .5))))}')
    print(f'99% Confidence Interval: {tuple(round(x, 4) for x in stats.norm.interval(.98, loc=backtest_mean, scale=(backtest_std/(len(df) ** .5))))}')
    print("-" * 50)

    if backtest:
        plt.hist(df['Model Result'], bins=50, density=True, color='blue', alpha=0.5, label='Model Result')
        plt.xlim(backtest_mini, backtest_maxi)
        plt.plot(backtest_overlay, backtest_p, 'blue', label='Model PDF')
        plt.axvline(backtest_mean, color='blue', linestyle='dashed', label='Model Mean')
        plt.text(backtest_mean, plt.ylim()[1] * .9, f'{round(backtest_mean, 2)}%', color='black', ha='center')
        plt.title(f'Normal Distribution of Model Results')
        if buy_hold is False and spy is False:
            #Standard Deviation Plots
            plt.axvline(backtest_mean + backtest_std, color='green', linestyle='dashed')
            plt.axvline(backtest_mean - backtest_std, color='green', linestyle='dashed')
            plt.axvline(backtest_mean + 2 * backtest_std, color='purple', linestyle='dashed')
            plt.axvline(backtest_mean - 2 * backtest_std, color='purple', linestyle='dashed')

            #labels
            plt.text(backtest_mean + backtest_std, plt.ylim()[1] * .8, '+1std', color='green', ha='center')
            plt.text(backtest_mean + 2 * backtest_std, plt.ylim()[1] * .7, '+2std', color='purple', ha='center')
            plt.text(backtest_mean - 2 * backtest_std, plt.ylim()[1] * .7, '-2std', color='purple', ha='center')
            plt.text(backtest_mean - backtest_std, plt.ylim()[1] * .8, '-1std', color='green', ha='center')

            #Confidence Intervals


        print("Model Results:")
        print(f"Model Mean: {round(backtest_mean, 2)}%")
        print(f"Model Std: {round(backtest_std, 2)}%")
    
    if buy_hold:
        plt.hist(df['Buy/Hold Result'], bins=50, density=True, color='orange', alpha=0.5, label='Buy/Hold Result')
        plt.xlim(buy_hold_mini, buy_hold_maxi)
        plt.plot(buy_hold_overlay, buy_hold_p, 'orange', label='Buy/Hold PDF')
        plt.axvline(buy_hold_mean, color='orange', linestyle='dashed', label='Buy/Hold Mean')
        plt.text(buy_hold_mean, plt.ylim()[1] * .9, f'{round(buy_hold_mean, 2)}%', color='black', ha='center')
        if backtest is False and spy is False:
            #Standard Deviation Plots
            plt.axvline(buy_hold_mean + buy_hold_std, color='green', linestyle='dashed')
            plt.axvline(buy_hold_mean - buy_hold_std, color='green', linestyle='dashed')
            plt.axvline(buy_hold_mean + 2 * buy_hold_std, color='purple', linestyle='dashed')
            plt.axvline(buy_hold_mean - 2 * buy_hold_std, color='purple', linestyle='dashed')

            #labels

            plt.text(buy_hold_mean + buy_hold_std, plt.ylim()[1] * .8, '+1std', color='green', ha='center')
            plt.text(buy_hold_mean + 2 * buy_hold_std, plt.ylim()[1] * .7, '+2std', color='purple', ha='center')
            plt.text(buy_hold_mean - 2 * buy_hold_std, plt.ylim()[1] * .7, '-2std', color='purple', ha='center')
            plt.text(buy_hold_mean - buy_hold_std, plt.ylim()[1] * .8, '-1std', color='green', ha='center')

        print("-" * 50)
        print("Buy/Hold Results:")
        print(f"Buy/Hold Mean: {round(buy_hold_mean, 2)}%")
        print(f"Buy/Hold Std: {round(buy_hold_std, 2)}%")

    if spy and 'SPY Buy/Hold Result' in df.columns:
        plt.hist(df['SPY Buy/Hold Result'], bins=50, density=True, color='green', alpha=0.5, label='SPY Buy/Hold Result')
        plt.xlim(spy_buy_hold_mini, spy_buy_hold_maxi)
        plt.plot(spy_buy_hold_overlay, spy_buy_hold_p, 'g', label='SPY Buy/Hold PDF')
        plt.axvline(spy_buy_hold_mean, color='green', linestyle='dashed', label='SPY Buy/Hold Mean')
        plt.text(spy_buy_hold_mean, plt.ylim()[1] * .9, f'{round(spy_buy_hold_mean, 2)}%', color='black', ha='center')
        if backtest is False and buy_hold is False:
            #Standard Deviation Plots
            plt.axvline(spy_buy_hold_mean + spy_buy_hold_std, color='blue', linestyle='dashed')
            plt.axvline(spy_buy_hold_mean - spy_buy_hold_std, color='blue', linestyle='dashed')
            plt.axvline(spy_buy_hold_mean + 2 * spy_buy_hold_std, color='purple', linestyle='dashed')
            plt.axvline(spy_buy_hold_mean - 2 * spy_buy_hold_std, color='purple', linestyle='dashed')

            #labels

            plt.text(spy_buy_hold_mean + spy_buy_hold_std, plt.ylim()[1] * .8, '+1std', color='blue', ha='center')
            plt.text(spy_buy_hold_mean + 2 * spy_buy_hold_std, plt.ylim()[1] * .7, '+2std', color='purple', ha='center')
            plt.text(spy_buy_hold_mean - 2 * spy_buy_hold_std, plt.ylim()[1] * .7, '-2std', color='purple', ha='center')
            plt.text(spy_buy_hold_mean - spy_buy_hold_std, plt.ylim()[1] * .8, '-1std', color='blue', ha='center')

        print("-" * 50)
        print("SPY Buy/Hold Results:")
        print(f"SPY Buy/Hold Mean: {round(spy_buy_hold_mean, 2)}%")
        print(f"SPY Buy/Hold Std: {round(spy_buy_hold_std, 2)}%")
    if backtest is True and buy_hold is True and spy is True and spy and 'SPY Buy/Hold Result' in df.columns:
        plt.title(f'Normal Distribution of Model and Buy/Hold Results')
        plt.xlim(min(backtest_mini, buy_hold_mini, spy_buy_hold_mini), max(backtest_maxi, buy_hold_maxi, spy_buy_hold_maxi))
    if backtest is True and buy_hold is True and spy is False:
        plt.title(f'Normal Distribution of Model and Buy/Hold Results')
        plt.xlim(min(backtest_mini, buy_hold_mini), max(backtest_maxi, buy_hold_maxi))
    if backtest is True and spy is True and buy_hold is False and 'SPY Buy/Hold Result' in df.columns:
        plt.title(f'Normal Distribution of Model and SPY Buy/Hold Results')
        plt.xlim(min(backtest_mini, spy_buy_hold_mini), max(backtest_maxi, spy_buy_hold_maxi))
    if buy_hold is True and spy is True and backtest is False and 'SPY Buy/Hold Result' in df.columns:
        plt.title(f'Normal Distribution of Asset Buy/Hold and SPY Buy/Hold Results')
        plt.xlim(min(buy_hold_mini, spy_buy_hold_mini), max(buy_hold_maxi, spy_buy_hold_maxi))

    plt.xlabel('Return (%)')
    plt.ylabel('Density')
    plt.legend()

    print("-" * 50)

def sharpe_hist(input, backtest=True, buy_hold=False, spy=False, csv_off=False):
    fig1 = plt.figure(figsize=(12, 6))
    if csv_off is False:
        df = pd.read_csv(input)
    if csv_off is True:
        df = input
    if df.empty:
        print("CSV file is empty. Please run the backtest first.")
        return

    backtest_mini, backtest_maxi = df['Model Sharpe'].min(), df['Model Sharpe'].max()
    backtest_mean = df['Model Sharpe'].mean()
    backtest_std = df['Model Sharpe'].std()
    backtest_overlay = np.linspace(backtest_mini, backtest_maxi, 100)
    backtest_p = norm.pdf(backtest_overlay, backtest_mean, backtest_std)
    if 'Buy/Hold Sharpe' in df.columns:
        buy_hold_mini, buy_hold_maxi = df['Buy/Hold Sharpe'].min(), df['Buy/Hold Sharpe'].max()
        buy_hold_mean = df['Buy/Hold Sharpe'].mean()
        buy_hold_std = df['Buy/Hold Sharpe'].std()
        buy_hold_overlay = np.linspace(buy_hold_mini, buy_hold_maxi, 100)
        buy_hold_p = norm.pdf(buy_hold_overlay, buy_hold_mean, buy_hold_std)
    if 'SPY Sharpe' in df.columns:
        spy_buy_hold_mini, spy_buy_hold_maxi = df['SPY Sharpe'].min(), df['SPY Sharpe'].max() 
        spy_buy_hold_mean = df['SPY Sharpe'].mean()
        spy_buy_hold_std = df['SPY Sharpe'].std()
        spy_buy_hold_overlay = np.linspace(spy_buy_hold_mini, spy_buy_hold_maxi, 100)
        spy_buy_hold_p = norm.pdf(spy_buy_hold_overlay, spy_buy_hold_mean, spy_buy_hold_std)

    print("-" * 50)
    print(f"n = {len(df)}")
    print("-" * 50)
    print(f'90% Confidence Interval: {tuple(round(x, 4) for x in stats.norm.interval(.90, loc=backtest_mean, scale=(backtest_std/(len(df) ** .5))))}')
    print(f'95% Confidence Interval: {tuple(round(x, 4) for x in stats.norm.interval(.95, loc=backtest_mean, scale=(backtest_std/(len(df) ** .5))))}')
    print(f'98% Confidence Interval: {tuple(round(x, 4) for x in stats.norm.interval(.98, loc=backtest_mean, scale=(backtest_std/(len(df) ** .5))))}')
    print(f'99% Confidence Interval: {tuple(round(x, 4) for x in stats.norm.interval(.98, loc=backtest_mean, scale=(backtest_std/(len(df) ** .5))))}')
    print("-" * 50)

    if backtest:
        plt.hist(df['Model Sharpe'], bins=50, density=True, color='blue', alpha=0.5, label='Model Sharpe')
        plt.xlim(backtest_mini, backtest_maxi)
        plt.plot(backtest_overlay, backtest_p, 'blue', label='Model PDF')
        plt.axvline(backtest_mean, color='blue', linestyle='dashed', label='Model Mean')
        plt.text(backtest_mean, plt.ylim()[1] * .9, f'{round(backtest_mean, 2)}', color='black', ha='center')
        plt.title('Normal Distribution of Model Sharpe')
        if buy_hold is False and spy is False:
            #Standard Deviation Plots
            plt.axvline(backtest_mean + backtest_std, color='green', linestyle='dashed')
            plt.axvline(backtest_mean - backtest_std, color='green', linestyle='dashed')
            plt.axvline(backtest_mean + 2 * backtest_std, color='purple', linestyle='dashed')
            plt.axvline(backtest_mean - 2 * backtest_std, color='purple', linestyle='dashed')

            #labels
            plt.text(backtest_mean + backtest_std, plt.ylim()[1] * .8, '+1std', color='green', ha='center')
            plt.text(backtest_mean + 2 * backtest_std, plt.ylim()[1] * .7, '+2std', color='purple', ha='center')
            plt.text(backtest_mean - 2 * backtest_std, plt.ylim()[1] * .7, '-2std', color='purple', ha='center')
            plt.text(backtest_mean - backtest_std, plt.ylim()[1] * .8, '-1std', color='green', ha='center')

        print("Model Sharpe:")
        print(f"Model Mean: {round(backtest_mean, 4)}")
        print(f"Model Std: {round(backtest_std, 4)}")
    
    if buy_hold:
        plt.hist(df['Buy/Hold Sharpe'], bins=50, density=True, color='orange', alpha=0.5, label='Buy/Hold Sharpe')
        plt.xlim(buy_hold_mini, buy_hold_maxi)
        plt.plot(buy_hold_overlay, buy_hold_p, 'orange', label='Buy/Hold PDF')
        plt.axvline(buy_hold_mean, color='orange', linestyle='dashed', label='Buy/Hold Mean')
        plt.text(buy_hold_mean, plt.ylim()[1] * .9, f'{round(buy_hold_mean, 2)}', color='black', ha='center')
        if backtest is False and spy is False:
            #Standard Deviation Plots
            plt.axvline(buy_hold_mean + buy_hold_std, color='green', linestyle='dashed')
            plt.axvline(buy_hold_mean - buy_hold_std, color='green', linestyle='dashed')
            plt.axvline(buy_hold_mean + 2 * buy_hold_std, color='purple', linestyle='dashed')
            plt.axvline(buy_hold_mean - 2 * buy_hold_std, color='purple', linestyle='dashed')

            #labels

            plt.text(buy_hold_mean + buy_hold_std, plt.ylim()[1] * .8, '+1std', color='green', ha='center')
            plt.text(buy_hold_mean + 2 * buy_hold_std, plt.ylim()[1] * .7, '+2std', color='purple', ha='center')
            plt.text(buy_hold_mean - 2 * buy_hold_std, plt.ylim()[1] * .7, '-2std', color='purple', ha='center')
            plt.text(buy_hold_mean - buy_hold_std, plt.ylim()[1] * .8, '-1std', color='green', ha='center')

        print("-" * 50)
        print("Buy/Hold Sharpe:")
        print(f"Buy/Hold Mean: {round(buy_hold_mean, 4)}")
        print(f"Buy/Hold Std: {round(buy_hold_std, 4)}")


        plt.title('Normal Distribution of Model and Buy/Hold Sharpe')
        plt.xlim(min(backtest_mini, buy_hold_mini), max(backtest_maxi, buy_hold_maxi))
    
    if spy and 'SPY Sharpe' in df.columns:
        plt.hist(df['SPY Sharpe'], bins=50, density=True, color='green', alpha=0.5, label='SPY Sharpe')
        plt.xlim(spy_buy_hold_mini, spy_buy_hold_maxi)
        plt.plot(spy_buy_hold_overlay, spy_buy_hold_p, 'green', label='SPY Sharpe PDF')
        plt.axvline(spy_buy_hold_mean, color='green', linestyle='dashed', label='SPY Mean')
        plt.text(spy_buy_hold_mean, plt.ylim()[1] * .9, f'{round(spy_buy_hold_mean, 2)}', color='black', ha='center')
        if backtest is False and buy_hold is False:
            #Standard Deviation Plots
            plt.axvline(spy_buy_hold_mean + spy_buy_hold_std, color='orange', linestyle='dashed')
            plt.axvline(spy_buy_hold_mean - spy_buy_hold_std, color='orange', linestyle='dashed')
            plt.axvline(spy_buy_hold_mean + 2 * spy_buy_hold_std, color='purple', linestyle='dashed')
            plt.axvline(spy_buy_hold_mean - 2 * spy_buy_hold_std, color='purple', linestyle='dashed')

            #labels
            plt.text(spy_buy_hold_mean + spy_buy_hold_std, plt.ylim()[1] * .8, '+1std', color='orange', ha='center')
            plt.text(spy_buy_hold_mean + 2 * spy_buy_hold_std, plt.ylim()[1] * .7, '+2std', color='purple', ha='center')
            plt.text(spy_buy_hold_mean - 2 * spy_buy_hold_std, plt.ylim()[1] * .7, '-2std', color='purple', ha='center')
            plt.text(spy_buy_hold_mean - spy_buy_hold_std, plt.ylim()[1] * .8, '-1std', color='orange', ha='center')

        print("-" * 50)
        print("SPY Sharpe:")
        print(f"SPY Mean: {round(spy_buy_hold_mean, 4)}")
        print(f"SPY Std: {round(spy_buy_hold_std, 4)}")

    if backtest is True and buy_hold is True and spy is True and 'SPY Sharpe' in df.columns:
        plt.title(f'Normal Distribution of Model and Buy/Hold Results')
        plt.xlim(min(backtest_mini, buy_hold_mini, spy_buy_hold_mini), max(backtest_maxi, buy_hold_maxi, spy_buy_hold_maxi))
    if backtest is True and buy_hold is True and spy is False:
        plt.title(f'Normal Distribution of Model and Buy/Hold Results')
        plt.xlim(min(backtest_mini, buy_hold_mini), max(backtest_maxi, buy_hold_maxi))
    if backtest is True and spy is True and buy_hold is False and 'SPY Sharpe' in df.columns:
        plt.title(f'Normal Distribution of Model and SPY Buy/Hold Results')
        plt.xlim(min(backtest_mini, spy_buy_hold_mini), max(backtest_maxi, spy_buy_hold_maxi))
    if buy_hold is True and spy is True and backtest is False and 'SPY Sharpe' in df.columns:
        plt.title(f'Normal Distribution of Asset Buy/Hold and SPY Buy/Hold Results')
        plt.xlim(min(buy_hold_mini, spy_buy_hold_mini), max(buy_hold_maxi, spy_buy_hold_maxi))

    plt.xlabel('Sharpe')
    plt.ylabel('Density')
    plt.legend()

    print("-" * 50)

def bear_hist(input, backtest=True, buy_hold=False, spy=False):
    # Answers the question of how did the model do when either the buy_hold or SPY were negative.
    # Grab all rows where backtest or spy were negative, then run functions
    df = pd.read_csv(input)
    if spy is True and buy_hold is True and 'SPY Buy/Hold Result' in df.columns:
        negative_both = df[(df['SPY Buy/Hold Result'] < 0) & (df['Buy/Hold Result'] < 0)]
        return_hist(negative_both, buy_hold=True, spy=True, csv_off=True)
        sharpe_hist(negative_both, buy_hold=True, spy=True, csv_off=True)
    if spy is True and buy_hold is False and 'SPY Buy/Hold Result' in df.columns:
        print("-" * 50)
        print("How Model Performs while SPY is down")
        print("-" * 50)
        negative_spy = df[df['SPY Buy/Hold Result'] < 0]
        return_hist(negative_spy, spy=True, csv_off=True)
        sharpe_hist(negative_spy, spy=True, csv_off=True)
    if buy_hold is True and spy is False:
        print("-" * 50)
        print("How Model Performs while asset is down")
        print("-" * 50)
        negative_buy_hold = df[df['Buy/Hold Result'] < 0]
        return_hist(negative_buy_hold, buy_hold=True, csv_off=True)
        sharpe_hist(negative_buy_hold, buy_hold=True, csv_off=True)
    if buy_hold is True and spy is True and 'SPY Buy/Hold Result' not in df.columns:
        print("-" * 50)
        print("How Model Performs while asset is down")
        print("-" * 50)
        negative_buy_hold = df[df['Buy/Hold Result'] < 0]
        return_hist(negative_buy_hold, buy_hold=True, csv_off=True)
        sharpe_hist(negative_buy_hold, buy_hold=True, csv_off=True)

def bull_hist(input, backtest=True, buy_hold=False, spy=False):
    # Answers the question of how did the model do when either the buy_hold or SPY were positive.
    # Grab all rows where backtest or spy were negative, then run functions
    df = pd.read_csv(input)
    if spy is True and buy_hold is True and 'SPY Buy/Hold Result' in df.columns:
        positive_both = df[(df['SPY Buy/Hold Result'] > 0) & (df['Buy/Hold Result'] > 0)]
        return_hist(positive_both, buy_hold=True, spy=True, csv_off=True)
        sharpe_hist(positive_both, buy_hold=True, spy=True, csv_off=True)
    if spy is True and buy_hold is False and 'SPY Buy/Hold Result' in df.columns:
        print("-" * 50)
        print("How Model Performs while SPY is up")
        print("-" * 50)
        positive_spy = df[df['SPY Buy/Hold Result'] > 0]
        return_hist(positive_spy, spy=True, csv_off=True)
        sharpe_hist(positive_spy, spy=True, csv_off=True)
    if buy_hold is True and spy is False:
        print("-" * 50)
        print("How Model Performs while asset is up")
        print("-" * 50)
        positive_buy_hold = df[df['Buy/Hold Result'] > 0]
        return_hist(positive_buy_hold, buy_hold=True, csv_off=True)
        sharpe_hist(positive_buy_hold, buy_hold=True, csv_off=True)
    if buy_hold is True and spy is True and 'SPY Buy/Hold Result' not in df.columns:
        # check if were working on SPY
        print("-" * 50)
        print("How Model Performs while asset is up")
        print("-" * 50)
        positive_buy_hold = df[df['Buy/Hold Result'] > 0]
        return_hist(positive_buy_hold, buy_hold=True, csv_off=True)
        sharpe_hist(positive_buy_hold, buy_hold=True, csv_off=True)

def neutral_hist(input, backtest=True, buy_hold=False, spy=False):
    # Answers the question of how did the model do when either the buy_hold or SPY were positive.
    # Grab all rows where backtest or spy were negative, then run functions
    df = pd.read_csv(input)
    if spy is True and buy_hold is True and 'SPY Buy/Hold Result' in df.columns:
        neutral_both = df[(df['SPY Buy/Hold Result'] > 0) & (df['Buy/Hold Result'] > 0)]
        return_hist(neutral_both, buy_hold=True, spy=True, csv_off=True)
        sharpe_hist(neutral_both, buy_hold=True, spy=True, csv_off=True)
    if spy is True and buy_hold is False and 'SPY Buy/Hold Result' in df.columns:
        print("-" * 50)
        print("How Model Performs while SPY is up")
        print("-" * 50)
        neutral_spy = df[df['SPY Buy/Hold Result'] > 0]
        return_hist(neutral_spy, spy=True, csv_off=True)
        sharpe_hist(neutral_spy, spy=True, csv_off=True)
    if buy_hold is True and spy is False:
        print("-" * 50)
        print("How Model Performs while asset is up")
        print("-" * 50)
        neutral_buy_hold = df[df['Buy/Hold Result'] > 0]
        return_hist(neutral_buy_hold, buy_hold=True, csv_off=True)
        sharpe_hist(neutral_buy_hold, buy_hold=True, csv_off=True)
    if buy_hold is True and spy is True and 'SPY Buy/Hold Result' not in df.columns:
        print("-" * 50)
        print("How Model Performs while asset is up")
        print("-" * 50)
        neutral_buy_hold = df[df['Buy/Hold Result'] > 0]
        return_hist(neutral_buy_hold, buy_hold=True, csv_off=True)
        sharpe_hist(neutral_buy_hold, buy_hold=True, csv_off=True)