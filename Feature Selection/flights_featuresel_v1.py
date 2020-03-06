import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def run_univariate(x, y, columns, n=3):

    test = SelectKBest(score_func=chi2, k=3)
    uni = test.fit(x, y)

    print()
    print('\033[1m' + 'Univariate Analysis' '\033[0m')

    # Automate column outputs for univariate analysis

    cols = uni.scores_.argsort()[-n:]
    print("The top features are: " + columns[cols[0]] +
          ", " + columns[cols[1]] + ", " + columns[cols[2]])
    features = uni.transform(x)
    print(features[0:5, :])


def run_rfe(x, y, columns, n=3):

    # Run Logistic Regression

    model = LogisticRegression(solver='liblinear')
    rfe = RFE(model, n)  # we want to find the 3 top features
    rec = rfe.fit(shan_x, shan_y)

    # Automate column outputs for RFE

    top_cols = col_list(rec.ranking_, columns)

    print('\033[1m' + 'Recursive Feature Elimination' '\033[0m')
    print()
    print(f'Number of features {rec.n_features_:d}')
    print(f'Selected features {rec.support_}')
    print(f'Ranking of features {rec.ranking_}')
    print()
    for x in range(n):
        print(str(x) + " Feature: " + str(top_cols[0]))

# Automating column printing function for recursive feature elimination


def col_list(ranking, column_names):
    var_s = ranking
    top_cols = []
    counter = 0
    for x in var_s:
        if var_s[counter] == 1:
            top_cols.append(column_names[counter])
        counter += 1
    return top_cols
