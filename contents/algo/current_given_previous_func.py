import pandas as pd
import numpy as np

# Probability of a security returning a certain percentage given the previous period's return.

# - - - Takes in a single df and returns the conditional probability

def conditional_probability(df, print_statement=True):
    # - - - Conditional Probability Setup - - -
    df = df.copy()

    # -- initialize ranges and labels for binning
    ranges = [-np.inf, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, np.inf]
    labels = ["<-8%", "-8% to -7%", "-7% to -6%", "-6% to -5%", "-5% to -4%", "-4% to -3%", "-3% to -2%", "-2% to -1%", "-1% to 0%", "0% to 1%", "1% to 2%", "2% to 3%", "3% to 4%", "4% to 5%", "5% to 6%", "6% to 7%", "7% to 8%", ">8%"]
    # -- create bins for previous and current returns
    df['Prev Close'] = df['Close'].shift(1)
    df["Previous Period Return"] = df["Return"].shift(1,fill_value=0)
    df["Previous Bin"] = pd.cut(df['Previous Period Return'], bins=ranges, labels=labels)
    df["Current Bin"] = pd.cut(df['Return'], bins=ranges, labels=labels)

    # -- create probability and count dataframes, with labels as both index and columns
    prob_df = pd.DataFrame(index=labels, columns=labels)
    count_df = pd.DataFrame(index=labels, columns=labels)

    # -- calculate counts and probabilities
    # for each combination of previous and current bins, calculate the count and probability
    for previous_bin in labels:
        for current_bin in labels:
            # Count how many times the previous bin and current bin occur together
            count_both = len(df[(df["Previous Bin"] == previous_bin) & (df["Current Bin"] == current_bin)])
            # Count how many times only the previous bin occurs
            count_prev = len(df[df["Previous Bin"] == previous_bin])
            # Store the counts and probabilities in the respective dataframes
            # count is used to calculate the probability
            count_df.loc[previous_bin, current_bin] = count_both
            probability = count_both / count_prev if count_prev > 0 else 0
            prob_df.loc[previous_bin, current_bin] = probability
    count_df = count_df.astype(int)
    prob_df = prob_df.astype(float)

    #- - - Format Columns - - -
    negative_return = ["<-8%", "-8% to -7%", "-7% to -6%", "-6% to -5%", "-5% to -4%", "-4% to -3%", "-3% to -2%", "-2% to -1%", "-1% to 0%"]
    positive_return = ["0% to 1%", "1% to 2%", "2% to 3%", "3% to 4%", "4% to 5%", "5% to 6%", "6% to 7%", "7% to 8%", ">8%"]
    prob_df["Negative"] = prob_df[negative_return].sum(axis=1)
    prob_df["Positive"] = prob_df[positive_return].sum(axis=1)
    count_df["Negative"] = count_df[negative_return].sum(axis=1)
    count_df["Positive"] = count_df[positive_return].sum(axis=1)

    prob_df[">1%"] = prob_df['1% to 2%'] + prob_df["2% to 3%"] + prob_df["3% to 4%"] + prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
    prob_df[">2%"] = prob_df["2% to 3%"] + prob_df["3% to 4%"] + prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
    prob_df[">3%"] = prob_df["3% to 4%"] + prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
    prob_df[">4%"] = prob_df["4% to 5%"] + prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
    prob_df[">5%"] = prob_df["5% to 6%"] + prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
    prob_df[">6%"] = prob_df["6% to 7%"] + prob_df["7% to 8%"] + prob_df[">8%"]
    prob_df[">7%"] = prob_df["7% to 8%"] + prob_df[">8%"]
    prob_df[">8%"] = prob_df[">8%"]
    # Calculate the Total Probability by summing coulmns
    #prob_df["Total"] = prob_df['Positive'] + prob_df['Negative']

    count_df[">1%"] = count_df['1% to 2%'] + count_df["2% to 3%"] + count_df["3% to 4%"] + count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
    count_df[">2%"] = count_df["2% to 3%"] + count_df["3% to 4%"] + count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
    count_df[">3%"] = count_df["3% to 4%"] + count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
    count_df[">4%"] = count_df["4% to 5%"] + count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
    count_df[">5%"] = count_df["5% to 6%"] + count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
    count_df[">6%"] = count_df["6% to 7%"] + count_df["7% to 8%"] + count_df[">8%"]
    count_df[">7%"] = count_df["7% to 8%"] + count_df[">8%"]
    count_df[">8%"] = count_df[">8%"]

    # Calculate the Total Count by summing coulmns
    count_df["Total"] = count_df['Positive'] + count_df['Negative']
    if print_statement == True:
        # Top 5 Probabilities
        print('-'*60)
        print('Relevant Probabilities having occurred more than 10 times')
        print('-'*60)
        print('Top 5 Probabilities')
        top_probs = prob_df.stack().nlargest(999999)
        best_count = 0
        for prob in top_probs.index:
            if best_count == 5:
                break
            # if the count has happened more than 10 times, print it
            if count_df.loc[prob[0], prob[1]] > 10:
                best_count += 1
                print(f'P({prob[1]} | {prob[0]}) = {round(prob_df.loc[prob[0], prob[1]] * 100, 4)}% --- occurred {count_df.loc[prob[0], prob[1]]} times')
        print('-'*60)

        # Top 5 Worst Probabilities excluding 0
        print('-'*60)
        print('Top 5 Worst Probabilities')
        worst_probs = prob_df.stack().nsmallest(999999)
        worst_count = 0
        for prob in worst_probs.index:
            if worst_count == 5:
                break
            if prob_df.loc[prob[0], prob[1]] > 0:
                # if the count has happened more than 10 times, print it
                if count_df.loc[prob[0], prob[1]] > 10:
                    worst_count += 1
                    print(f'P({prob[1]} | {prob[0]}) = {round(prob_df.loc[prob[0], prob[1]] * 100, 4)}% --- occurred {count_df.loc[prob[0], prob[1]]} times')
    if df['Return'].count() < 100:
        raise Exception("Not enough data: fewer than 100 samples. Conditional probability table will not be returned. Try a date that would allow for more samples.")
    return prob_df  