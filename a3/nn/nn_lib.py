import time
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
from nn import preprocess_nn as pp


def solve():
    df = pd.read_csv(
        'poker_dataset\poker-hand-training-true.data')
    X_train = df.iloc[:, :-1].values
    y_train = df.iloc[:, -1].values
    df = None

    try:
        clf = pickle.load(open('nn_models/poker_model.pkl', 'rb'))
    except:

        clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=(100, 100),  activation='relu')
        print("Training")
        t1 = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        print("Training time:", t2-t1)
        # save using pickle
        with open('nn_models/poker_model.pkl', 'wb') as f:
            pickle.dump(clf, f)

    X_train = None
    y_train = None
    print("Predicting")
    df = pd.read_csv(
        'poker_dataset/poker-hand-testing.data')
    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values
    print(clf.score(X_test, y_test))
    print(clf.predict(X_test))


def solve2():
    train_path = "./poker_dataset/poker-hand-training-true-onehot.data"
    test_path = "./poker_dataset/poker-hand-testing-onehot.data"
    train_data = pp.get_data(train_path)
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    try:
        clf = pickle.load(open('nn_models/poker_model.pkl', 'rb'))
    except:

        clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=(100, 100),  activation='relu')
        print("Training")
        t1 = time.time()
        clf.fit(X_train, y_train)
        t2 = time.time()
        print("Training time:", t2-t1)
        # save using pickle
        with open('nn_models/poker_model.pkl', 'wb') as f:
            pickle.dump(clf, f)

    X_train = None
    y_train = None
    print("Predicting")
    test_path = pp.get_data(test_path)
    X_test = test_path.iloc[:, :-1]
    y_test = test_path.iloc[:, -1]
    print(clf.score(X_test, y_test))
    print(clf.predict(X_test))
