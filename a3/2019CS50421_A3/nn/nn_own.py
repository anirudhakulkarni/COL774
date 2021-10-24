# import preprocess_nn as pn
import pandas as pd
from datetime import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import keyboard
import pickle
import numpy as np
import time
# import preprocess_nn as pn
from numpy.lib.scimath import sqrt
np.random.seed(10)


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
    plt.savefig("nn_results/confusion_matrix"+label+".png")
    plt.close()


def activation(a, type):
    if type == 'sigmoid':
        return 1 / (1 + np.exp(-a))
    elif type == 'tanh':
        return np.tanh(a)
    elif type == 'relu':
        return np.maximum(0, a)


class perceptrons():
    '''
    wT*x+b = y
    perceptron class is collection of perceptrons in the layer
    arguments:
        input_feature_size: size of input vector, i.e. number of perceptrons in the previous layer
        perceptron_count: number of perceptrons in the layer
        type: layer type, i.e. input, hidden, output
    '''

    def __init__(self, input_feature_size, perceptron_count, type, activation_type, m):
        self.weights = np.random.uniform(-1/sqrt(input_feature_size), 1/sqrt(input_feature_size),
                                         (input_feature_size, perceptron_count))
        self.type = type
        self.perceptron_count = perceptron_count
        self.input_feature_size = input_feature_size
        self.gradients = np.zeros((input_feature_size, perceptron_count))
        self.output = None
        self.activation_type = activation_type
        self.netj = None
        self.input = None
        self.m = m
        self.b = np.zeros((perceptron_count, 1))

    def calculate_delta(self,  perceptrons_next=None, target=None):
        if self.type == 'output':
            if self.activation_type == 'sigmoid':
                self.delta = np.multiply(np.multiply(
                    target - self.output, self.output), (1 - self.output))
            elif self.activation_type == 'relu':
                self.delta = np.multiply(target - self.output, self.netj > 0)
            self.gradients = - np.dot(self.input, self.delta.T)/self.m
        elif self.type == 'hidden':
            if self.activation_type == 'sigmoid':
                self.delta = np.multiply(np.multiply(np.dot(
                    perceptrons_next.weights, perceptrons_next.delta), self.output), (1 - self.output))
            elif self.activation_type == 'relu':
                self.delta = np.multiply(np.dot(
                    perceptrons_next.weights, perceptrons_next.delta), self.netj > 0)
            self.gradients = - np.dot(self.input, self.delta.T)/self.m
        else:
            raise Exception('Invalid layer type')
        return self.gradients

    def update_weights(self,  learning_rate, epoch=None):
        if epoch is not None:
            self.weights = self.weights - learning_rate / \
                sqrt(epoch) * self.gradients
            self.b = self.b + (learning_rate/sqrt(epoch)) * \
                (np.sum(self.delta, axis=1)).reshape(-1, 1)/self.m
        else:
            self.weights = self.weights - learning_rate * self.gradients
            self.b = self.b + learning_rate * \
                (np.sum(self.delta, axis=1)).reshape(-1, 1)/self.m

    def predict(self, input_vector):
        self.netj = np.dot(self.weights.T, input_vector)
        self.netj = self.netj + self.b
        self.output = activation(self.netj, self.activation_type)
        return self.output


class NeuralNetwork():
    '''
    Assumptions:
        1. input_vector is (features_num, example_num)
        2. label is (target_num, example_num)

    '''

    def __init__(self, features_num, hidden_number, hidden_size, target_num, activation_type, m, adaptive=False):
        self.features_num = features_num
        self.hidden_number = hidden_number
        self.hidden_size = hidden_size
        self.target_num = target_num
        self.hidden_layers = [perceptrons(
            features_num, hidden_size[0], "hidden", activation_type, m)]
        self.hidden_layers += [perceptrons(hidden_size[i-1], hidden_size[i], "hidden", activation_type, m)
                               for i in range(1, hidden_number)]
        self.output_layer = perceptrons(
            hidden_size[-1], target_num, "output", 'sigmoid', m)
        self.activation_type = activation_type
        self.adaptive = adaptive
        self.train_time = 0

    def train(self, input_vector, label, learn_rate, epochs, batch_size, epsilon=0.001):
        label_mat = np.zeros((self.target_num, label.shape[0]))
        for i in range(label.shape[0]):
            label_mat[int(label[i])][i] = 1
        label_mat = np.matrix(label_mat)
        label = label_mat
        last_error = -1e9
        time_start = time.time()
        for i in range(epochs):
            print("epoch: ", i)
            for j in range(0, input_vector.shape[1], batch_size):
                label_mat_batch = label[:, j:j+batch_size]
                input_vector_batch = input_vector.iloc[:, j:j+batch_size]
                self.forward_pass(input_vector_batch)
                self.output_layer.calculate_delta(target=label_mat_batch)
                for h in range(self.hidden_number-1, -1, -1):
                    if h == self.hidden_number-1:
                        self.hidden_layers[h].calculate_delta(
                            self.output_layer)
                    else:
                        self.hidden_layers[h].calculate_delta(
                            self.hidden_layers[h+1])
                if self.adaptive:
                    self.output_layer.update_weights(learn_rate, i+1)
                    for h in range(self.hidden_number-1, -1, -1):
                        self.hidden_layers[h].update_weights(learn_rate, i+1)

                else:
                    self.output_layer.update_weights(learn_rate)
                    for h in range(self.hidden_number-1, -1, -1):
                        self.hidden_layers[h].update_weights(learn_rate)
                # self.forward_pass(input_vector_batch)
                # error = np.sum(
                #     np.square(label_mat_batch - self.output_layer.output))/input_vector_batch.shape[1]
                # if error < epsilon or abs(last_error-error) < epsilon:
                #     self.train_time = time.time() - time_start
                #     return
                # last_error = error
            self.forward_pass(input_vector)
            error = np.sum(
                np.square(label_mat - self.output_layer.output))/input_vector.shape[1]
            if error < epsilon or abs(last_error-error) < epsilon:
                self.train_time = time.time() - time_start
                return
            last_error = error

            # break when ctr+* is pressed
            if keyboard.is_pressed('ctrl+*'):
                self.train_time = time.time() - time_start
                return
            print("Error:", error)
        self.train_time = time.time() - time_start

    def forward_pass(self, input_vector):
        last_out = None
        for h in self.hidden_layers:
            if last_out is None:
                h.input = input_vector
                last_out = h.predict(input_vector)
            else:
                h.input = last_out
                last_out = h.predict(last_out)
        self.output_layer.input = last_out
        return self.output_layer.predict(last_out)

    def test(self, input_vector, label, name):
        prediction = self.forward_pass(input_vector)
        prediction = np.array([np.argmax(i) for i in prediction.T])
        outs = np.count_nonzero(prediction == label)
        accu = outs/label.shape[0]
        # confusion matrix and f1 score
        conf_mat = confusion_matrix(label, prediction)
        f1 = f1_score(label, prediction, average='weighted')
        print("confusion matrix: ", conf_mat)
        print("f1 score: ", f1)
        print("accuracy: ", accu)
        draw_confusion(conf_mat, name)
        # append confusion matrix and f1 score to file
        with open("nn_results/"+name+".txt", 'a') as f:
            f.write(name + '\n')
            f.write("Train time: "+str(self.train_time) + '\n')
            f.write(str(conf_mat) + '\n')
            f.write("F1 score: "+str(f1) + '\n')
            f.write("Accuracy: "+str(outs/label.shape[0]) + '\n')
        return accu


# if __name__ == "__main__":
#     train_data = pd.read_csv("../test.csv")
#     features_num = len(train_data.iloc[0])-1
#     hidden_size = [2]
#     hidden_number = len(hidden_size)
#     target_num = 2
#     epochs = 1
#     batch_size = 5
#     learn_rate = 0.1
#     epsilon = 1e-9
#     nn = NeuralNetwork(features_num=features_num, hidden_number=hidden_number, hidden_size=hidden_size,
#                        target_num=target_num, activation_type='sigmoid', m=batch_size, adaptive=False)
#     nn.train(train_data.iloc[:, :-1].T, train_data.iloc[:, -1],
#              learn_rate, epochs, batch_size, epsilon)
# if __name__ == '__main__':
#     train_data = pn.get_data(
#         '..\poker_dataset\poker-hand-training-true-onehot.data')
#     features_num = len(train_data.iloc[0])-1
#     hidden_size = [5]
#     hidden_number = len(hidden_size)

#     target_num = 10
#     epochs = 600
#     batch_size = 100
#     learn_rate = 0.1
#     epsilon = 1e-9
#     try:
#         # read pickle model
#         nn = pickle.load(open('nn-5-sigm-best.pkl', 'rb'))

#         nn.train(train_data.iloc[:, :-1].T,
#                  train_data.iloc[:, -1], learn_rate, epochs, batch_size, epsilon)
#         # pickle.dump(nn, open('nn-100-relu.pkl', 'wb'))
#         # nn = pickle.load(open('nn-100-relu.pkl', 'rb'))
#         pickle.dump(nn, open('nn-5-sigm-best.pkl', 'wb'))
#         print("model saved")
#     except:
#         activation_type = 'sigmoid'
#         train_data = pn.get_data(
#             '..\poker_dataset\poker-hand-training-true-onehot.data')
#         nn = NeuralNetwork(features_num=features_num, hidden_number=hidden_number, hidden_size=hidden_size,
#                            target_num=target_num, activation_type=activation_type, m=batch_size, adaptive=False)

#         nn.train(train_data.iloc[:, :-1].T,
#                  train_data.iloc[:, -1], learn_rate, epochs, batch_size, epsilon)
#         # pickle.dump(nn, open('nn-100-relu.pkl', 'wb'))
#         pickle.dump(nn, open('nn-5-sigm-best.pkl', 'wb'))
#         print("model saved")
#     # nn = None
#     # nn = pickle.load(open('nn-100-relu.pkl', 'rb'))
#     # nn = pickle.load(open('nn-15-sigm-best.pkl', 'rb'))
#     train_data = None
#     test_data = pn.get_data('..\poker_dataset\poker-hand-testing-onehot.data')
#     nn.test(test_data.iloc[:, :-1].T, test_data.iloc[:, -1])
#     test_data = None
#     test_data = pn.get_data(
#         '..\poker_dataset\poker-hand-training-true-onehot.data')
#     nn.test(test_data.iloc[:, :-1].T, test_data.iloc[:, -1])
