from os import sep
from numpy.lib.function_base import median
import preprocess_dt as ppd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# global variables
index=0
# constants
class node:
    # multiple nodes can be created
    def __init__(self, index,subdata,attribute,isLeaf=False,target=None):
        self.index = index
        self.subdata = subdata
        self.attribute = attribute
        self.isLeaf = isLeaf
        self.children = []
        if isLeaf :
            self.create_leaf(target)
    
    def __str__(self):
        if self.isLeaf:
            return "Leaf node: "+str(self.target)
        else:
            return ""
            # return "Node: "+str(self.attribute)+" "+str(self.children)

    def add_child(self, child,value,symbol):
        self.children.append([child,value,symbol])
    def create_leaf(self, target):
        self.target = target
        self.children = []
    # destructor
    # def __del__(self):
    #     del self.index
    #     del self.target
    #     del self.attribute
    #     del self.children
class DataSet():
    def __init__(self, data):
        self.data = data
        self.target = data.iloc[:,-1]
        self.len=len(data)
        # self.attributes = attributes
    # destructor
    def __del__(self):
        del self.data
        del self.target
        del self.attributes

class DecisionTree():
    def __init__(self,train_path="bank_dataset/bank_train.csv",test_path="bank_dataset/bank_test.csv",validation_path="bank_dataset/bank_val.csv",onehot=False):
        self.train_data,self.test_data,self.validation_data = self.get_data(train_path,test_path,validation_path,onehot)
        self.root=None
        self.train_pred=None
        self.test_pred=None
        self.validation_pred=None
        # self.train_data=DataSet(self.train_data)
        # self.test_data=DataSet(self.test_data)
        # self.validation_data=DataSet(self.validation_data)

    def get_data(self,train_path,test_path,validation_path,onehot):
        if onehot:
            train_data = pd.read_csv(train_path, sep=';')
            test_data = pd.read_csv(test_path, sep=';')
            validation_data = pd.read_csv(validation_path,sep=';')
            train_data = ppd.one_hot_encoding(train_data)
            test_data = ppd.one_hot_encoding(test_data)
            validation_data = ppd.one_hot_encoding(validation_data)
        else:
            train_data = pd.read_csv(train_path, sep=';')
            test_data = pd.read_csv(test_path, sep=';')
            validation_data = pd.read_csv(validation_path,sep=';')
        return train_data,test_data,validation_data
        
    def train(self):
        self.root = self._build_tree(self.train_data,5)
        pass
    
    def save(self):
        pass
    def load(self):
        pass
    def evaluate(self):
        pass

    # private functions
    def _build_tree(self, data, max_depth=None,n_features=None):
        # Build a decision tree
        global index
        if len(data) == 0:
            print("Empty data")
            return None
        if max_depth is not None and max_depth <= 0:
            print("Leaf Node",data.iloc[:,-1].value_counts().idxmax())
            return node(index,data,None,True,data.iloc[:,-1].value_counts().idxmax())
        if len(np.unique(data.iloc[:,-1])) == 1:
            print("**************************************************")
            print("Leaf Node",data.iloc[:,-1].value_counts().idxmax())
            # print(data)
            return node(index,data,None,True,data.iloc[:,-1].value_counts().idxmax())
        if n_features is None:
            n_features = len(data.iloc[0]) - 1

        # split
        best_feature = self._choose_best_feature(data, n_features)
        tree=node(index,data,best_feature);index+=1

        print(data[best_feature])
        if data[best_feature].dtype == 'object':
            feature_values = np.unique(data[best_feature])
            for value in feature_values:
                sub_data = data.loc[data[best_feature] == value]
                sub_tree = self._build_tree(sub_data, max_depth - 1, n_features)
                tree.add_child(sub_tree,value,"=")
            
        else:
            median = self._get_categories(data, best_feature,'numerical')
            sub_data1 = data.loc[data[best_feature] <= median]
            sub_data2 = data.loc[data[best_feature] > median]
            sub_tree1 = self._build_tree(sub_data1, max_depth - 1, n_features)
            sub_tree2 = self._build_tree(sub_data2, max_depth - 1, n_features)
            tree.add_child(sub_tree1,median,"<=")
            tree.add_child(sub_tree2,median,">")
        return tree

    def _choose_best_feature(self, data, n_features):
        # Choose the best feature to split the data
        # best attribute is which minimizes conditional entropy
        best_feature = None
        best_mutual_information = None
        for feature in data.columns:
            if feature == 'target':
                continue
            if data[feature].dtype == 'object':
                feature_values = np.unique(data[feature])
                temp=0
                for value in feature_values:
                    sub_data = data.loc[data[feature] == value]
                    y_1=len(sub_data.loc[sub_data.iloc[:,-1]==1])
                    y_0=len(sub_data)-y_1
                    frac=y_1/len(sub_data)
                    temp+=len(sub_data)/len(data)*(-1)*(frac*np.log2(frac)+((1-frac)*np.log2(1-frac)))
                if best_mutual_information is None or temp < best_mutual_information:
                    best_mutual_information = temp
                    best_feature = feature
            else:
                median = self._get_categories(data, feature, 'numerical')
                sub_data1 = data.loc[data[feature] <= median]
                sub_data2 = data.loc[data[feature] > median]
                y_1_1=len(sub_data1.loc[sub_data1.iloc[:,-1]==1])
                y_0_1=len(sub_data1)-y_1_1
                y_1_2=len(sub_data2.loc[sub_data2.iloc[:,-1]==1])
                y_0_2=len(sub_data2)-y_1_2
                try:
                    frac_1=y_1_1/len(sub_data1)
                    frac_2=y_1_2/len(sub_data2)
                    temp=len(sub_data1)/len(data)*(-1)*(frac_1*np.log2(frac_1)+((1-frac_1)*np.log2(1-frac_1)))+len(sub_data2)/len(data)*(-1)*(frac_2*np.log2(frac_2)+((1-frac_2)*np.log2(1-frac_2)))
                    if best_mutual_information is None or temp < best_mutual_information:
                        best_mutual_information = temp
                        best_feature = feature
                except:
                    continue
        return best_feature
    def test(self):
        # self.test_data is data set
        self.test_data['pred']=None
        print(self.root)

        print(self._predict(self.test_data,self.root))
        print(self.test_data['pred'])
        print(self.test_data['y'])
        # get accuracy
        acc=0
        for i in range(len(self.test_data)):
            if self.test_data.loc['pred',i]==self.test_data.loc['y',i]:
                acc+=1
        acc=acc/len(self.test_data)
        pass
    def _predict(self,test_data,root):
        # predict the target
        print(root)
        print("NON leaf")
        if root is None:
            print("root is None")
            return test_data
        if root.isLeaf:
            print("LEAF")
            # select indices in self.test_data based on test_data
            print(test_data.index)
            test_data['pred']=root.target
            
            # test_data['pred']=root.target
        else:
            for child in root.children:
                if child[2]=="=":
                    test_data=self._predict(test_data.loc[test_data[root.attribute] == child[1]],child[0])
                else:
                    if child[2] == "<=":
                        test_data=self._predict(test_data[root.attribute] <= child[1],child[0])
                    else:
                        test_data=self._predict(test_data[root.attribute] > child[1],child[0])
        return test_data

    def _entropy(self, target,type):
        # Calculate the entropy
        entropy = 0
        if type == 'classification':
            classes = np.unique(target)
            for c in classes:
                p = len(target.loc[target == c]) / len(target)
                entropy += p * np.log2(p)
        elif type=='numerical':
            median=np.median(target)
            p1=len(target.loc[target<=median])/len(target)
            p2=len(target.loc[target>median])/len(target)
            entropy=-p1*np.log2(p1)-p2*np.log2(p2)
        return -entropy
    
    def _conditional_entropy(self, data, target, feature, type='classification'):
        # Calculate the conditional entropy between the target and the feature
        if type == 'classification':
            feature_values = np.unique(data[feature])
            conditional_entropy = 0
            for value in feature_values:
                sub_target = target.loc[data[feature] == value]
                conditional_entropy += len(sub_target) / len(target) * self._entropy(sub_target)
        elif type == 'numerical':
            median = np.median(data[feature])
            sub_target1 = target.loc[data[feature] <= median]
            sub_target2 = target.loc[data[feature] > median]
            conditional_entropy = len(sub_target1) / len(target) * self._entropy(sub_target1) + \
                                  len(sub_target2) / len(target) * self._entropy(sub_target2)
        return conditional_entropy
    
    def _mutual_information(self, data,target,feature,type='classification'):
        return self._entropy(target,type) - self._conditional_entropy(data,target,feature,type)

    def _get_categories(self, data, feature,type):
        # Get the categories of the feature
        if type == 'classification':
            categories = np.unique(data[feature])
        elif type == 'numerical':
            median=np.median(data[feature])
            categories = [median]

if __name__ == '__main__':
    dt = DecisionTree(onehot=True)
    # dt.get_data(train_path=)
    dt.train()
    dt.test()
    dt.evaluate()


