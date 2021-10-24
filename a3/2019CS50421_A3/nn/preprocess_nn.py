import pandas as pd
import numpy as np
categories_list = ['suit_1', 'rank_1', 'suit_2', 'rank_2', 'suit_3',
                   'rank_3', 'suit_4', 'rank_4', 'suit_5', 'rank_5', 'ordinal']
# [suit_1_1,suit_1_2,suit_1_3,suit_1_4,rank_1_10,rank_1_11,rank_1_12,rank_1_1,rank_1_2,rank_1_9,rank_1_5,rank_1_6,rank_1_13,rank_1_8,rank_1_3,rank_1_4,rank_1_7,suit_2_1,suit_2_2,suit_2_3,suit_2_4,rank_2_11,rank_2_13,rank_2_4,rank_2_12,rank_2_2,rank_2_6,rank_2_1,rank_2_7,rank_2_8,rank_2_10,rank_2_5,rank_2_9,rank_2_3,suit_3_1,suit_3_2,suit_3_3,suit_3_4,rank_3_13,rank_3_10,rank_3_1,rank_3_12,rank_3_5,rank_3_3,rank_3_9,rank_3_2,rank_3_4,rank_3_7,rank_3_6,rank_3_11,rank_3_8,suit_4_1,suit_4_2,suit_4_3,suit_4_4,rank_4_12,rank_4_10,rank_4_13,rank_4_11,rank_4_3,rank_4_4,rank_4_7,rank_4_5,rank_4_1,rank_4_2,rank_4_6,rank_4_9,rank_4_8,suit_5_1,suit_5_2,suit_5_3,suit_5_4,rank_5_1,rank_5_12,rank_5_10,rank_5_6,rank_5_13,rank_5_5,rank_5_8,rank_5_3,rank_5_9,rank_5_11,rank_5_2,rank_5_7,rank_5_4,ordinal]
# [suit_1_1,suit_1_2,suit_1_3,suit_1_4,rank_1_10,rank_1_11,rank_1_12,rank_1_1,rank_1_2,rank_1_9,rank_1_5,rank_1_6,rank_1_13,rank_1_8,rank_1_3,rank_1_4,rank_1_7,suit_2_1,suit_2_2,suit_2_3,suit_2_4,rank_2_11,rank_2_13,rank_2_4,rank_2_12,rank_2_2,rank_2_6,rank_2_1,rank_2_7,rank_2_8,rank_2_10,rank_2_5,rank_2_9,rank_2_3,suit_3_1,suit_3_2,suit_3_3,suit_3_4,rank_3_13,rank_3_10,rank_3_1,rank_3_12,rank_3_5,rank_3_3,rank_3_9,rank_3_2,rank_3_4,rank_3_7,rank_3_6,rank_3_11,rank_3_8,suit_4_1,suit_4_2,suit_4_3,suit_4_4,rank_4_12,rank_4_10,rank_4_13,rank_4_11,rank_4_3,rank_4_4,rank_4_7,rank_4_5,rank_4_1,rank_4_2,rank_4_6,rank_4_9,rank_4_8,suit_5_1,suit_5_2,suit_5_3,suit_5_4,rank_5_1,rank_5_12,rank_5_10,rank_5_6,rank_5_13,rank_5_5,rank_5_8,rank_5_3,rank_5_9,rank_5_11,rank_5_2,rank_5_7,rank_5_4,ordinal]
final_colmn = []


def read_data(filename='./dataset/train.csv'):
    dataset = pd.read_csv(filename, sep=',', header=None)
    # print(dataset.head())
    # X=dataset.iloc[:,:-1]
    # Y=dataset.iloc[:,-1]
    return dataset


def get_data(filename='poker_dataset\poker-hand-testing-onehot.data'):
    try:
        return pd.read_csv(filename+".clean", sep=',')
    except:
        dataset = pd.read_csv(filename, sep=',')
        for col in range(len(dataset.columns)-2):
            dataset.iloc[:, col] = (dataset.iloc[:, col] -
                                    dataset.iloc[:, col].mean())/dataset.iloc[:, col].std()
        print(dataset.head())
        # save data
        pd.DataFrame(dataset).to_csv(filename+".clean", index=False)
        return dataset
# one hot encoding


def one_hot_encoding(X):
    global final_colmn
    # create dataframe and add header
    X_one_hot = pd.DataFrame(columns=final_colmn)
    for cat in range(len(X.columns)-1):
        # select rows with given category in column cat
        # choose only categorial data
        categories = X[cat].unique()
        print(categories)
        for category in categories:
            # create a new column for each category
            X_one_hot[categories_list[cat]+'_' +
                      str(category)] = np.where(X[cat] == category, 1, 0)
    X_one_hot['ordinal'] = X.iloc[:, -1]
    final_colmn = X_one_hot.columns
    return X_one_hot

# save file as csv


def save_data(X, filename='./dataset/train_processed.csv'):
    df = pd.DataFrame(X)
    df.to_csv(filename, index=False)
