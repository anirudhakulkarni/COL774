import pandas as pd
import numpy as np
categories_list=['suit_1','rank_1','suit_2','rank_2','suit_3','rank_3','suit_4','rank_4','suit_5','rank_5','ordinal']
def read_data(filename='./dataset/train.csv'):
    dataset = pd.read_csv(filename,sep=',',header=None)
    # print(dataset.head())
    # X=dataset.iloc[:,:-1]
    # Y=dataset.iloc[:,-1]
    return dataset
def get_data(filename='poker_dataset\poker-hand-testing-onehot.data'):
    dataset = pd.read_csv(filename,sep=',')
    return dataset.T
# one hot encoding
def one_hot_encoding(X):
    # create dataframe and add header
    X_one_hot=pd.DataFrame()
    for cat in range(len(X.columns)-1):
        # select rows with given category in column cat
        # choose only categorial data
        categories=X[cat].unique()
        print(categories)
        for category in categories:
            # create a new column for each category
            X_one_hot[categories_list[cat]+'_'+str(category)]=np.where(X[cat]==category,1,0)
    X_one_hot['ordinal']=X.iloc[:,-1]
    return X_one_hot

# save file as csv
def save_data(X,filename='./dataset/train_processed.csv'):
    df=pd.DataFrame(X)
    df.to_csv(filename,index=False)