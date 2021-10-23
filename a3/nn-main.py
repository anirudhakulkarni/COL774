import sys
from nn import preprocess_nn as pp
import numpy as np
from nn import nn_own
import matplotlib.pyplot as plt
import pickle

# main function for neural network
if __name__ == '__main__':
    train_path = "./poker_dataset/poker-hand-training-true-onehot.data"
    test_path = "./poker_dataset/poker-hand-testing-onehot.data"

    target_num = 10
    epochs = 500
    batch_size = 100
    learn_rate = 0.1
    epsilon = 1e-9
    features_num = 85

    # take arguments
    question = sys.argv[1]

    if question == '1':
        train_path = "./poker_dataset/poker-hand-training-true.data"
        test_path = "./poker_dataset/poker-hand-testing.data"
        data_train = pp.read_data(train_path)
        data_test = pp.read_data(test_path)
        data_train = pp.one_hot_encoding(data_train)
        data_test = pp.one_hot_encoding(data_test)
        print(data_test)
        # save as csv
        pp.save_data(
            data_train, "./poker_dataset/poker-hand-training-true-onehot.data")
        pp.save_data(
            data_test, "./poker_dataset/poker-hand-testing-onehot.data")
    elif question == '3':

        units_in_layer = [5, 10, 15, 20, 25]
        f1_score_list = []
        for unit in units_in_layer:
            nn = None
            nn = nn_own.NeuralNetwork(features_num=features_num, hidden_number=1, hidden_size=[unit],
                                      target_num=target_num, activation_type='sigmoid', m=batch_size, adaptive=False)
            train_data = pp.get_data(train_path)
            nn.train(train_data.iloc[:, :-1].T,
                     train_data.iloc[:, -1], learn_rate, epochs, batch_size, epsilon)
            nn.test(train_data.iloc[:, :-1].T,
                    train_data.iloc[:, -1], str(unit)+"_train_qc")
            train_data = None
            test_data = pp.get_data(test_path)
            f1_score_list.append(nn.test(test_data.iloc[:, :-1].T,
                                         test_data.iloc[:, -1], str(unit)+"_test_qc"))
            test_data = None
            pickle.dump(nn, open(str(unit)+"_nn_qc.pkl", "wb"))
        # plot f1 score vs units
        print(f1_score_list)
        plt.figure()
        plt.plot(units_in_layer, f1_score_list)
        plt.xlabel('units')
        plt.ylabel('f1 score')
        plt.savefig('f1_score_vs_units-qc.png')
        plt.show()

    elif question == '4':

        units_in_layer = [5, 10, 15, 20, 25]
        f1_score_list = []
        for unit in units_in_layer:
            nn = None
            nn = nn_own.NeuralNetwork(features_num=features_num, hidden_number=1, hidden_size=[unit],
                                      target_num=target_num, activation_type='sigmoid', m=batch_size, adaptive=True)
            train_data = pp.get_data(train_path)
            nn.train(train_data.iloc[:, :-1].T,
                     train_data.iloc[:, -1], learn_rate, epochs, batch_size, epsilon)
            nn.test(train_data.iloc[:, :-1].T,
                    train_data.iloc[:, -1], str(unit)+"_train_qd")
            train_data = None
            test_data = pp.get_data(test_path)
            f1_score_list.append(nn.test(test_data.iloc[:, :-1].T,
                                         test_data.iloc[:, -1], str(unit)+"_test_qd"))
            test_data = None
            pickle.dump(nn, open(str(unit)+"_nn_qd.pkl", "wb"))
        # plot f1 score vs units
        print(f1_score_list)
        plt.figure()
        plt.plot(units_in_layer, f1_score_list)
        plt.xlabel('units')
        plt.ylabel('f1 score')
        plt.savefig('f1_score_vs_units-qd.png')
        plt.show()
    elif question == '5':
        