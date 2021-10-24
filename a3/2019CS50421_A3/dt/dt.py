import sys
from dt import preprocess_dt as ppd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import csv
from sklearn.metrics import confusion_matrix, f1_score

index = 1
depth = 6
onehot = False


def draw_confusion(conf, label="linear"):
    plt.figure()
    plt.imshow(conf)
    plt.title("Confusion Matrix"+label)
    plt.colorbar()
    my_xticks = [i for i in range(len(conf))]
    plt.xticks(my_xticks, my_xticks)
    my_yticks = [i for i in range(len(conf))]
    plt.yticks(my_yticks, my_yticks)
    plt.set_cmap("Greens")
    plt.ylabel("True labels")
    plt.xlabel("Predicted label")
    # add points on the axis
    for i in range(len(conf)):
        for j in range(len(conf)):
            plt.text(j, i, str(conf[i, j]), ha="center",
                     va="center", color="black")
    plt.savefig("confusion_matrix"+label+".png")
    plt.close()


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


class node:
    # multiple nodes can be created
    def __init__(self, index, subdata, attribute, isLeaf=False, target=None):
        self.index = index
        self.len_subdata = len(subdata)  # ???
        self.attribute = attribute
        self.isLeaf = isLeaf
        self.children = []
        self.val_data = None
        self.target = subdata.iloc[:, -1].value_counts().idxmax()
        if isLeaf:
            self.create_leaf(target)
        print("************************************************")
        print(self)
        # print(len(self.target))

    def __str__(self):
        if self.isLeaf:
            return "Leaf node: "+str(self.index)+str(self.target)
        else:
            # return ""
            return "Node number: "+str(self.index)+", attribute: "+str(self.attribute)+" "+str(self.children)

    def add_child(self, child, value, symbol):
        self.children.append([child, value, symbol])

    def create_leaf(self, target):
        self.target = target
        self.children = []


class DecisionTree():
    def __init__(self, train_path="bank_dataset/bank_train.csv", onehot=False, depth=0):
        self.train_path = train_path
        self.validation_path = "bank_dataset/bank_val.csv"
        self.test_path = "bank_dataset/bank_test.csv"
        self.train_data = self.get_data(train_path, onehot)
        self.root = None
        self.totalNodes = 1
        self.node_count = 1
        self.onehot = onehot
        self.depth = depth
        self.prunecounter = 0

    def set_data(self, test_path="bank_dataset/bank_train.csv", validation_path="bank_dataset/bank_val.csv", onehot=False):
        self.test_data = self.get_data(test_path, onehot)
        self.validation_data = self.get_data(validation_path, onehot)
        self.test_path = test_path

    def set_test(self, test_path):
        self.test_data = self.get_data(test_path, self.onehot)

    def get_data(self, train_path, onehot):

        if onehot:
            train_data = pd.read_csv(train_path, sep=';')
            print("ONE HOT ENABLED")
            train_data = ppd.one_hot_encoding(train_data)
        else:
            train_data = pd.read_csv(train_path, sep=';')
        return train_data

    def count_nodes(self):
        self.node_count = self._count_nodes(self.root)
        return self.node_count

    def _count_nodes(self, root):
        if root.isLeaf:
            return 1
        else:
            count = 1
            for child in root.children:
                count += self._count_nodes(child[0])
            return count

    def train(self):
        self.root = self._build_tree(self.train_data, self.depth)

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self):
        pass

    # private functions
    def _build_tree(self, data, max_depth):
        # Build a decision tree
        global index
        if len(data) == 0:
            print("Empty data")
            return None
        if max_depth is not None and max_depth <= 0:
            print("Leaf Node due to max depth:",
                  data.iloc[:, -1].value_counts())
            return node(index, data, None, True, data.iloc[:, -1].value_counts().idxmax())
        if len(np.unique(data.iloc[:, -1])) == 1:
            print("Leaf Node as all elements classified:",
                  data.iloc[:, -1].value_counts())
            return node(index, data, None, True, data.iloc[:, -1].value_counts().idxmax())

        # split
        best_feature = self._choose_best_feature(data)
        print("Best feature:", best_feature)
        tree = node(index, data, best_feature)
        index += 1
        self.totalNodes += 1
        if data[best_feature].dtype == 'object':
            feature_values = np.unique(data[best_feature])
            for value in feature_values:
                sub_data = data.loc[data[best_feature] == value]
                sub_tree = self._build_tree(
                    sub_data, max_depth - 1)
                if sub_tree is not None:
                    tree.add_child(sub_tree, value, "=")

        else:
            median = np.median(data[best_feature])
            sub_data1 = data.loc[data[best_feature] <= median]
            sub_data2 = data.loc[data[best_feature] > median]
            sub_tree1 = self._build_tree(sub_data1, max_depth - 1)
            sub_tree2 = self._build_tree(sub_data2, max_depth - 1)
            if sub_tree1 is not None:
                tree.add_child(sub_tree1, median, "<=")
            if sub_tree2 is not None:
                tree.add_child(sub_tree2, median, ">")
        return tree

    def prune(self, prune_list=None):
        self.validation_data['pred'] = 'no'
        # self.validation_data = self.validation_data.loc[self.validation_data.index]
        self._prune_predict(self.validation_data, self.root, True)
        print("Validation data:", self.validation_data)
        if prune_list:
            self.prune_list = prune_list
            self.prune_test_list = []
            self.prune_train_list = []
            self.prune_val_list = []
            self.prunecounter = 0
            self.validation_path = "bank_dataset/bank_val.csv"
            self.test_path = "bank_dataset/bank_test.csv"
            self.train_path = "bank_dataset/bank_train.csv"
            self.prune_tree_withlist(self.root)
        else:
            self.prune_tree(self.root)

    def prune_tree(self, root):
        if self.max_prune <= 0:
            return
        if root.isLeaf:
            return
        if len(root.children) == 1:
            return
        for child in root.children:
            self.prune_tree(child[0])
            if self.max_prune <= 0:
                return
            old_preds = self.validation_data.loc[child[0].val_data.index]['pred']
            old_ys = self.validation_data.loc[child[0].val_data.index]['y']
            old_acc = len(old_preds[old_preds == old_ys])
            new_acc = len(old_ys[old_ys == child[0].target])
            incr_acc = new_acc-old_acc
            if incr_acc > 0:
                self.max_prune -= 1
                # print("Pruning node:", child.index)
                child[0].isLeaf = True
                self.validation_data.loc[child[0].val_data.index]['pred'] = child[0].target

        return

    def prune_tree_withlist(self, root):
        if root.isLeaf:
            return
        if len(root.children) == 1:
            return
        for child in root.children:
            self.prune_tree_withlist(child[0])
            old_preds = self.validation_data.loc[child[0].val_data.index]['pred']
            old_ys = self.validation_data.loc[child[0].val_data.index]['y']
            old_acc = len(old_preds[old_preds == old_ys])
            new_acc = len(old_ys[old_ys == child[0].target])
            incr_acc = new_acc-old_acc
            if incr_acc > 0:
                self.prunecounter += 1
                child[0].isLeaf = True
                self.validation_data.loc[child[0].val_data.index]['pred'] = child[0].target
                if self.prunecounter in self.prune_list:
                    print(self.prunecounter)
                    self.set_test(self.test_path)
                    self.prune_test_list.append(self.test())
                    self.set_test(self.validation_path)
                    self.prune_val_list.append(self.test())
                    self.set_test(self.train_path)
                    self.prune_train_list.append(self.test())
                # print("Pruning node:", child.index)

        return

    def _choose_best_feature(self, data):
        # Choose the best feature to split the data
        # best attribute is which minimizes conditional entropy
        best_feature = None
        best_mutual_information = None
        for feature in data.columns:
            if feature == 'y':
                continue
            if data[feature].dtype == 'object':
                feature_values = np.unique(data[feature])
                temp = 0
                for value in feature_values:
                    sub_data = data.loc[data[feature] == value]
                    y_1 = len(sub_data.loc[sub_data.iloc[:, -1] == 'no'])
                    y_0 = len(sub_data)-y_1
                    if y_1 == 0 or y_0 == 0:
                        return feature
                    frac = y_1/len(sub_data)
                    temp += len(sub_data)/len(data)*(-1) * \
                        (frac*np.log2(frac)+((1-frac)*np.log2(1-frac)))
                if best_mutual_information is None or temp < best_mutual_information:
                    best_mutual_information = temp
                    best_feature = feature
            else:
                median = np.median(data[feature])
                sub_data1 = data.loc[data[feature] <= median]
                sub_data2 = data.loc[data[feature] > median]
                y_1_1 = len(sub_data1.loc[sub_data1.iloc[:, -1] == 'no'])
                y_0_1 = len(sub_data1)-y_1_1
                y_1_2 = len(sub_data2.loc[sub_data2.iloc[:, -1] == 'no'])
                y_0_2 = len(sub_data2)-y_1_2
                if len(sub_data2) == 0:
                    if y_1_1 == 0 or y_0_1 == 0:
                        return feature
                    frac = y_1_1/len(sub_data1)
                    temp = len(sub_data1)/len(data)*(-1) * \
                        (frac*np.log2(frac)+((1-frac)*np.log2(1-frac)))
                elif y_1_1 == 0 or y_0_1 == 0 or y_1_2 == 0 or y_0_2 == 0:
                    return feature
                else:
                    frac_1 = y_1_1/len(sub_data1)
                    frac_2 = y_1_2/len(sub_data2)
                    temp = len(sub_data1)/len(data)*(-1)*(frac_1*np.log2(frac_1)+((1-frac_1)*np.log2(1-frac_1)))+len(
                        sub_data2)/len(data)*(-1)*(frac_2*np.log2(frac_2)+((1-frac_2)*np.log2(1-frac_2)))
                if best_mutual_information is None or temp < best_mutual_information:
                    best_mutual_information = temp
                    best_feature = feature
        return best_feature

    def test(self):
        self.test_data['pred'] = 'no'
        self._predict(self.test_data, self.root)
        acc = 0
        for i in range(len(self.test_data)):
            if self.test_data.loc[i, 'pred'] == self.test_data.loc[i, 'y']:
                acc += 1
        acc = acc/len(self.test_data)
        # conf = confusion_matrix(self.test_data['y'], self.test_data['pred'])
        # f1 = f1_score(
        #     self.test_data['y'], self.test_data['pred'], average='weighted')
        # draw_confusion(conf, 'test')
        return acc
        # with open('dt_results/dt-results-'+str(self.onehot)+'.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([self.depth, self.totalNodes, onehot, self.test_path[18:-4],
        #                     conf[0][0], conf[0][1], conf[1][0], conf[1][1], acc,  f1])

    def _predict(self, test_data, root):
        # predict the target
        # print(root)
        if root is None:
            return
        if root.isLeaf:
            self.test_data.at[test_data.index, 'pred'] = root.target
        else:
            for child in root.children:
                if child[2] == "=":
                    self._predict(
                        test_data.loc[test_data[root.attribute] == child[1]], child[0])
                else:
                    if child[2] == "<=":
                        self._predict(
                            test_data.loc[test_data[root.attribute] <= child[1]], child[0])
                    else:
                        self._predict(
                            test_data.loc[test_data[root.attribute] > child[1]], child[0])
        return

    def _prune_predict(self, val_data, root, val=False):
        # print(self.validation_data)
        # print(self.validation_data.index)
        # print(self.validation_data.at[val_data.index, 'y'])
        # kk
        if root is None:
            return
        if root.isLeaf:
            root.val_data = val_data
            print("---------------------------------")
            # print(val_data.index)
            # print(self.validation_data.index)
            self.validation_data.at[root.val_data.index, 'pred'] = root.target
            # print(self.validation_data.loc[root.val_data.index]['pred'])

        else:
            for child in root.children:
                if child[2] == "=":
                    child[0].val_data = val_data.loc[val_data[root.attribute] == child[1]]
                    self._prune_predict(
                        child[0].val_data, child[0], val)
                else:
                    if child[2] == "<=":
                        child[0].val_data = val_data.loc[val_data[root.attribute] <= child[1]]
                        self._prune_predict(
                            child[0].val_data, child[0], val)
                    else:
                        child[0].val_data = val_data.loc[val_data[root.attribute] > child[1]]
                        self._prune_predict(
                            child[0].val_data, child[0], val)
        return

    def _entropy(self, target, type):
        # Calculate the entropy
        entropy = 0
        if type == 'classification':
            classes = np.unique(target)
            for c in classes:
                p = len(target.loc[target == c]) / len(target)
                entropy += p * np.log2(p)
        elif type == 'numerical':
            median = np.median(target)
            p1 = len(target.loc[target <= median])/len(target)
            p2 = len(target.loc[target > median])/len(target)
            entropy = -p1*np.log2(p1)-p2*np.log2(p2)
        return -entropy

    def _conditional_entropy(self, data, target, feature, type='classification'):
        # Calculate the conditional entropy between the target and the feature
        if type == 'classification':
            feature_values = np.unique(data[feature])
            conditional_entropy = 0
            for value in feature_values:
                sub_target = target.loc[data[feature] == value]
                conditional_entropy += len(sub_target) / \
                    len(target) * self._entropy(sub_target)
        elif type == 'numerical':
            median = np.median(data[feature])
            sub_target1 = target.loc[data[feature] <= median]
            sub_target2 = target.loc[data[feature] > median]
            conditional_entropy = len(sub_target1) / len(target) * self._entropy(sub_target1) + \
                len(sub_target2) / len(target) * self._entropy(sub_target2)
        return conditional_entropy

    def _mutual_information(self, data, target, feature, type='classification'):
        return self._entropy(target, type) - self._conditional_entropy(data, target, feature, type)

    def _get_categories(self, data, feature, type):
        # Get the categories of the feature
        if type == 'classification':
            categories = np.unique(data[feature])
        elif type == 'numerical':
            median = np.median(data[feature])
            categories = [median]


if __name__ == '__main__':
    depth = int(sys.argv[1])
    onehot = sys.argv[2] == "True"
    print(depth, onehot)
    try:
        with open('dt_models/dt-'+str(depth)+"-"+str(onehot)+'new.pkl', 'rb') as inp:
            dt = pickle.load(inp)
    except:
        dt = DecisionTree(onehot=onehot)
        dt.train()
        save_object(dt, 'dt_models/dt-'+str(depth)+"-"+str(onehot)+'new.pkl')
    dt.set_data(onehot=onehot, test_path="bank_dataset/bank_train.csv")
    dt.count_nodes()

    dt.prune()
    print(dt.node_count)
    dt.count_nodes()
    print(dt.node_count)

    dt.test()
    dt.set_data(onehot=onehot, test_path="bank_dataset/bank_test.csv")
    dt.test()
    dt.set_data(onehot=onehot, test_path="bank_dataset/bank_val.csv")
    dt.test()
    dt.evaluate()
