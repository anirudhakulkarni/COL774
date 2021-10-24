import sys
from nn import preprocess_nn as pp
import numpy as np
from nn import nn_own
import matplotlib.pyplot as plt
import pickle
from nn import nn_lib
# main function for neural network
if __name__ == '__main__':
    train_path = "./poker_dataset/poker-hand-training-true-onehot.data"
    test_path = "./poker_dataset/poker-hand-testing-onehot.data"

    target_num = 10
    epochs = 1400
    batch_size = 100
    learn_rate = 0.1
    epsilon = 1e-5
    features_num = 85

    # take arguments
    question = sys.argv[1]

    if question == 'a':
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

    elif question == 'c':
        units_in_layer = [5, 10, 15, 20, 25]
        f1_score_list = []
        train_acc_list = []
        test_acc_list = []
        time_taken_list = []
        for unit in units_in_layer:
            nn = None
            nn = nn_own.NeuralNetwork(features_num=features_num, hidden_number=1, hidden_size=[unit],
                                      target_num=target_num, activation_type='sigmoid', m=batch_size, adaptive=False)
            train_data = pp.get_data(train_path)
            nn.train(train_data.iloc[:, :-1].T,
                     train_data.iloc[:, -1], learn_rate, epochs, batch_size, epsilon)
            train_acc_list.append(nn.test(train_data.iloc[:, :-1].T,
                                          train_data.iloc[:, -1], str(unit)+"_train_qc"))
            train_data = None
            test_data = pp.get_data(test_path)
            test_acc_list.append(nn.test(test_data.iloc[:, :-1].T,
                                         test_data.iloc[:, -1], str(unit)+"_test_qc"))
            test_data = None
            # pickle.dump(nn, open(str(unit)+"_nn_qc.pkl", "wb"))
            time_taken_list.append(nn.train_time)
        print(f1_score_list)
        plt.figure()
        plt.plot(units_in_layer, train_acc_list, label="train accuracy")
        plt.plot(units_in_layer, test_acc_list, label="test accuracy")
        plt.xlabel("units in layer")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title('accuracy vs units')
        plt.savefig('nn_results/qc-accuracy_vs_units.png')
        plt.show()
        # plot time taken vs units
        plt.figure()
        plt.plot(units_in_layer, time_taken_list)
        plt.xlabel('units')
        plt.ylabel('time taken')
        plt.title('time taken vs units')
        plt.savefig('nn_results/qc-time_taken_vs_units.png')
        plt.show()

    elif question == 'd':
        target_num = 10
        epochs = 1400
        batch_size = 100
        learn_rate = 10
        epsilon = 1e-6
        features_num = 85
        units_in_layer = [5, 10, 15, 20, 25]
        train_acc_list = []
        test_acc_list = []
        time_taken_list = []
        for unit in units_in_layer:
            nn = None
            nn = nn_own.NeuralNetwork(features_num=features_num, hidden_number=1, hidden_size=[unit],
                                      target_num=target_num, activation_type='sigmoid', m=batch_size, adaptive=True)
            train_data = pp.get_data(train_path)
            nn.train(train_data.iloc[:, :-1].T,
                     train_data.iloc[:, -1], learn_rate, epochs, batch_size, epsilon)
            train_acc_list.append(nn.test(train_data.iloc[:, :-1].T,
                                          train_data.iloc[:, -1], str(unit)+"_train_qd"+str(learn_rate)))
            train_data = None
            test_data = pp.get_data(test_path)
            test_acc_list.append(nn.test(test_data.iloc[:, :-1].T,
                                         test_data.iloc[:, -1], str(unit)+"_test_qd"+str(learn_rate)))
            test_data = None
            # pickle.dump(nn, open(str(unit)+"_nn_qd.pkl", "wb"))
            time_taken_list.append(nn.train_time)
        # plot f1 score vs units
        print(train_acc_list)
        print(test_acc_list)
        plt.figure()
        plt.plot(units_in_layer, train_acc_list, label="train accuracy")
        plt.plot(units_in_layer, test_acc_list, label="test accuracy")
        plt.xlabel("units in layer")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title('Accuracy vs units adaptive learning rate '+str(learn_rate))
        plt.savefig('nn_results/qd-accu_vs_units_adaptive' +
                    str(learn_rate)+'.png')
        plt.show()
        # plot time taken vs units
        plt.figure()
        plt.plot(units_in_layer, time_taken_list)
        plt.xlabel('units')
        plt.ylabel('time taken')
        plt.title('time taken vs units'+str(learn_rate))
        plt.savefig('nn_results/qd-time_taken_vs_units'+str(learn_rate)+'.png')
        plt.show()
    elif question == 'e':
        learn_rate = 10
        nn = nn_own.NeuralNetwork(features_num=features_num, hidden_number=2, hidden_size=[
                                  100, 100], target_num=target_num, activation_type='sigmoid', m=batch_size, adaptive=True)
        train_data = pp.get_data(train_path)
        nn.train(train_data.iloc[:, :-1].T, train_data.iloc[:, -1],
                 learn_rate, epochs, batch_size, epsilon)
        acc_train_sigm = nn.test(train_data.iloc[:, :-1].T,
                                 train_data.iloc[:, -1], "train_sigm_qe")
        train_data = None
        test_data = pp.get_data(test_path)
        acc_test_sigm = nn.test(test_data.iloc[:, :-1].T,
                                test_data.iloc[:, -1], "test_sigm_qe")
        nn = None
        test_data = None

        nn = nn_own.NeuralNetwork(features_num=features_num, hidden_number=2, hidden_size=[
            100, 100], target_num=target_num, activation_type='relu', m=batch_size, adaptive=True)
        train_data = pp.get_data(train_path)
        nn.train(train_data.iloc[:, :-1].T, train_data.iloc[:, -1],
                 learn_rate, epochs, batch_size, epsilon)
        acc_train_relu = nn.test(train_data.iloc[:, :-1].T,
                                 train_data.iloc[:, -1], "train_relu_qe")
        train_data = None
        test_data = pp.get_data(test_path)
        acc_test_relu = nn.test(test_data.iloc[:, :-1].T,
                                test_data.iloc[:, -1], "test_relu_qe")
        nn = None
        test_data = None
        print("Train accuracy sigmoid:", acc_train_sigm)
        print("Test accuracy sigmoid:", acc_test_sigm)
        print("Train accuracy relu:", acc_train_relu)
        print("Test accuracy relu:", acc_test_relu)

    elif question == 'f':
        nn_lib.solve2()
