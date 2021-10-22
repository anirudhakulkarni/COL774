# random forest using sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    # read the data
    # one hot encoder

    train_data = pd.get_dummies(pd.read_csv(
        'bank_dataset/bank_train.csv', sep=';'))
    test_data = pd.get_dummies(pd.read_csv(
        'bank_dataset/bank_test.csv', sep=';'))
    val_data = pd.get_dummies(pd.read_csv(
        'bank_dataset/bank_val.csv', sep=';'))

    rf = RandomForestClassifier(
        criterion='entropy', bootstrap=True, oob_score=True)

    X_train, Y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, Y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
    X_val, Y_val = val_data.iloc[:, :-1], val_data.iloc[:, -1]

    # : (a) n estimators (50 to 450 in range of 100). (b) max features (0.1 to 1.0 in range of 0.2) (c) min samples split (2 to 10 in range of 2).
    # param_grid = {'n_estimators': [50, 150, 250, 350, 450], 'max_features': [
    #     0.1, 0.3, 0.5, 0.7, 0.9], 'min_samples_split': [2,  4, 6, 8, 10]}
    param_grid = {'n_estimators': [5], 'max_features': [
        0.1], 'min_samples_split': [2]}
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, Y_train)
    print('Best score: ', grid_search.best_score_)
    print('Best parameters: ', grid_search.best_params_)
    # print(rf.oob_score_)
    # print(rf.oob_decision_function_)
    rf2 = RandomForestClassifier(
        n_estimators=5, max_features=0.1, min_samples_split=2, criterion='entropy', bootstrap=True, oob_score=True)
    rf2.fit(X_train, Y_train)
    print(rf2.oob_score_)
    Y_pred = rf2.predict(X_test)
    print('Accuracy: ', accuracy_score(Y_test, Y_pred))
    Y_pred = rf2.predict(X_train)
    print('Accuracy: ', accuracy_score(Y_train, Y_pred))
