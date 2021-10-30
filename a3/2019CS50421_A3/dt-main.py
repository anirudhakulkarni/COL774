import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

from numpy.core import machar
from dt import dt
from dt import preprocess_dt as pp
from dt import dt_c_d as dtcd
# main function for neural network
if __name__ == '__main__':
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    val_path = sys.argv[3]
    question = sys.argv[4]
    subpart = sys.argv[5]
    print("start")
    if question == 'a':
        # load data
        tree_depths_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35]
        # tree_depths_list = [0, 2]
        if subpart == 'a':
            test_acc_list = []
            train_acc_list = []
            val_acc_list = []
            node_count_list = []
            for depth in tree_depths_list:
                print("depth:", depth)
                try:
                    with open('dt_models/dt-'+str(depth)+'-qa.pkl', 'rb') as inp:
                        mydt = pickle.load(inp)
                except:
                    mydt = dt.DecisionTree(
                        depth=depth, onehot=True, train_path=train_path)
                    mydt.train()
                    dt.save_object(mydt, 'dt_models/dt-'+str(depth)+'-qa.pkl')
                mydt.set_data(test_path=train_path,
                              validation_path=val_path, onehot=True)
                train_acc_list.append(mydt.test())
                mydt.set_data(test_path=test_path,
                              validation_path=val_path, onehot=True)
                test_acc_list.append(mydt.test())
                mydt.set_data(test_path=val_path,
                              validation_path=val_path, onehot=True)
                val_acc_list.append(mydt.test())
                node_count_list.append(mydt.count_nodes())
            mydt = None
            print(train_acc_list)
            print(test_acc_list)
            print(val_acc_list)
            print(node_count_list)
            plt.plot(node_count_list, train_acc_list,
                     'r', label='train', marker='o')
            plt.plot(node_count_list, test_acc_list,
                     'b', label='test', marker='o')
            plt.plot(node_count_list, val_acc_list,
                     'g', label='validation', marker='o')
            plt.xlabel('Node counts')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Node counts - Qa')
            plt.legend()
            plt.savefig('dt_results/dt-qa.png')
        # plt.show()
        elif subpart == 'b':
            # without onehot
            test_acc_list = []
            train_acc_list = []
            node_count_list = []
            val_acc_list = []
            for depth in tree_depths_list:
                print("depth:", depth)
                try:
                    with open('dt_models/dt-'+str(depth)+'-qa-mult.pkl', 'rb') as inp:
                        mydt = pickle.load(inp)
                except:
                    mydt = dt.DecisionTree(
                        depth=depth, onehot=False, train_path=train_path)
                    mydt.train()
                    dt.save_object(mydt, 'dt_models/dt-' +
                                   str(depth)+'-qa-mult.pkl')
                mydt.set_data(test_path=train_path,
                              validation_path=val_path, onehot=False)
                train_acc_list.append(mydt.test())
                mydt.set_data(test_path=test_path,
                              validation_path=val_path, onehot=False)
                test_acc_list.append(mydt.test())
                mydt.set_data(test_path=val_path,
                              validation_path=val_path, onehot=False)
                val_acc_list.append(mydt.test())
                node_count_list.append(mydt.count_nodes())
            mydt = None
            print(train_acc_list)
            print(test_acc_list)
            print(val_acc_list)
            print(node_count_list)
            # plot data with marker
            plt.figure()
            plt.plot(node_count_list, train_acc_list,
                     'r', label='train', marker='o')
            plt.plot(node_count_list, test_acc_list,
                     'b', label='test', marker='o')
            plt.plot(node_count_list, val_acc_list,
                     'g', label='validation', marker='o')
            plt.xlabel('Node counts')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Node counts - Qa-mult')
            plt.legend()
            plt.savefig('dt_results/dt-qa-mult.png')
    elif question == 'b':
        # pruning
        prune_max_list = [100, 400, 700, 1000, 1300, 1600]
        tree_depths_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35]
        # tree_depths_list = [50]
        if subpart == 'a':
            test_acc_list = []
            train_acc_list = []
            node_count_list = []
            depth = 35
            try:
                with open('dt_models/dt-'+str(depth)+'-qa.pkl', 'rb') as inp:
                    mydt = pickle.load(inp)
            except:
                mydt = dt.DecisionTree(
                    depth=depth, onehot=True, train_path=train_path)
                mydt.train()
                dt.save_object(mydt, 'dt_models/dt-'+str(depth)+'-qa.pkl')
            mydt.set_data(test_path=train_path,
                          validation_path=val_path, onehot=True)
            mydt.prune(prune_max_list)
            print(mydt.prune_train_list)
            print(mydt.prune_test_list)
            print(mydt.prune_val_list)
            # plot 3 lists against prune_max_list
            plt.figure()
            plt.plot(prune_max_list, mydt.prune_train_list,
                     'r', label='train', marker='o')
            plt.plot(prune_max_list, mydt.prune_test_list,
                     'b', label='test', marker='o')
            plt.plot(prune_max_list, mydt.prune_val_list,
                     'g', label='val', marker='o')
            plt.xlabel('Number of subtres pruned')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Number of subtres pruned - Qb')
            plt.legend()
            plt.savefig('dt_results/dt-qb-prune.png')
        if subpart == 'b':
            test_acc_list = []
            train_acc_list = []
            node_count_list = []
            prune_max_list = [100, 200, 300, 400, 500, 600, 700, 800]
            depth = 35
            try:
                with open('dt_models/dt-'+str(depth)+'-qa-mult.pkl', 'rb') as inp:
                    mydt = pickle.load(inp)
            except:
                mydt = dt.DecisionTree(
                    depth=depth, onehot=False, train_path=train_path)
                mydt.train()
                dt.save_object(mydt, 'dt_models/dt-'+str(depth)+'-qa-mult.pkl')
            mydt.set_data(test_path=train_path,
                          validation_path=val_path, onehot=False)
            mydt.prune(prune_max_list)
            print(mydt.prune_train_list)
            print(mydt.prune_test_list)
            print(mydt.prune_val_list)
            # plot 3 lists against prune_max_list
            plt.figure()
            plt.plot(prune_max_list, mydt.prune_train_list,
                     'r', label='train', marker='o')
            plt.plot(prune_max_list, mydt.prune_test_list,
                     'b', label='test', marker='o')
            plt.plot(prune_max_list, mydt.prune_val_list,
                     'g', label='val', marker='o')
            plt.xlabel('Number of subtres pruned')
            plt.ylabel('Accuracy')
            plt.title('Accuracy vs Number of subtres pruned - Qb')
            plt.legend()
            plt.savefig('dt_results/dt-qb-prune-mult.png')
    elif question == 'c':
        dtcd.part_c()
    elif question == 'd':
        dtcd.part_d()
