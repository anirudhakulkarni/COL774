import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
if __name__ == '__main__':
    df = pd.read_csv(
        'poker_dataset\poker-hand-training-true.data')
    X_train = df.iloc[:, :-1].values
    y_train = df.iloc[:, -1].values
    df = pd.read_csv(
        'poker_dataset/poker-hand-testing.data')
    X_test = df.iloc[:, :-1].values
    y_test = df.iloc[:, -1].values

    try:
        clf = pickle.load(open('poker_model110.pkl', 'rb'))
    except:

        clf = MLPClassifier(solver='sgd',
                            hidden_layer_sizes=(100, 100),  activation='relu')
        print("Fitting")
        clf.fit(X_train, y_train)
        # save using pickle
        with open('poker_model10.pkl', 'wb') as f:
            pickle.dump(clf, f)

    print("Predicting")
    print(clf.score(X_test, y_test))
    print(clf.predict(X_test))
