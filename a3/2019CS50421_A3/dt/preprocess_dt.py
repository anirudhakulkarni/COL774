from typing import final
import pandas as pd
import numpy as np
# categories_list=['suit_1','rank_1','suit_2','rank_2','suit_3','rank_3','suit_4','rank_4','suit_5','rank_5','ordinal']
final_colmn = []


def read_data(filename='./dataset/train.csv'):
    dataset = pd.read_csv(filename, sep=';')
    return dataset

# one hot encoding


def one_hot_encoding(X):
    # create dataframe and add header
    global final_colmn
    X_one_hot = pd.DataFrame(columns=final_colmn)
    categories_list = X.columns.tolist()
    for cat in range(len(X.columns)):

        if X.dtypes[cat] == 'object':
            categories = X.iloc[:, cat].unique()
            # print(categories)
            if categories.size == 2:
                X_one_hot[categories_list[cat]] = np.where(
                    X.iloc[:, cat] == categories_list[0], "yes", "no")
            else:
                for category in categories:
                    # create a new column for each category
                    X_one_hot[categories_list[cat]+'_'+str(category)] = np.where(
                        X.iloc[:, cat] == category, "yes", "no")
        else:
            X_one_hot[categories_list[cat]] = X.iloc[:, cat]
    X_one_hot['y'] = X.iloc[:, -1]
    final_colmn = X_one_hot.columns
    return X_one_hot

# save file as csv


def save_data(X, filename='./dataset/train_processed.csv'):
    df = pd.DataFrame(X)
    df.to_csv(filename, index=False)
