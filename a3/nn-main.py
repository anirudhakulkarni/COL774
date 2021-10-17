import sys
import preprocess_nn as pp
import numpy as np
import nn
# main function for neural network
if __name__ == '__main__':

    # take arguments
    question=sys.argv[1]
    
    if question == '1':
        train_path = "./poker_dataset/poker-hand-training-true.data"
        test_path = "./poker_dataset/poker-hand-testing.data"
        data_train= pp.read_data(train_path)
        data_test = pp.read_data(test_path)
        data_train=pp.one_hot_encoding(data_train)
        data_test=pp.one_hot_encoding(data_test)
        print(data_test)
        # save as csv
        pp.save_data(data_train, "./poker_dataset/poker-hand-training-true-onehot.data")
        pp.save_data(data_test, "./poker_dataset/poker-hand-testing-onehot.data")
    elif question =='2':
        train_path = "./poker_dataset/poker-hand-training-true-onehot.data"
        test_path = "./poker_dataset/poker-hand-testing-onehot.data"
        data_train= pp.read_data(train_path)
        data_test = pp.read_data(test_path)
        nn = nn.NeuralNetwork(features_num=85, hidden_num=10, labels_num=10)
        nn.train(data_train, epochs=100, learning_rate=0.1)
        nn.test(data_train)