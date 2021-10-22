import pickle
import numpy as np
import preprocess_nn as pn
from numpy.core.shape_base import hstack
from numpy.lib.scimath import sqrt
np.random.seed(10)


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

    def train(self, input_vector, label, learn_rate, epochs, batch_size):
        label_mat = np.zeros((self.target_num, label.shape[0]))
        for i in range(label.shape[0]):
            label_mat[int(label[i])][i] = 1
        label_mat = np.matrix(label_mat)
        label = label_mat
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
            self.forward_pass(input_vector)
            error = np.sum(
                np.square(label_mat - self.output_layer.output))/input_vector.shape[1]
            print(error)

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

    def test(self, input_vector, label):
        prediction = self.forward_pass(input_vector)
        prediction = np.array([np.argmax(i) for i in prediction.T])
        np.savetxt('prediction.csv', prediction, delimiter=',')
        outs = np.count_nonzero(prediction == label)
        print(outs/label.shape[0])
        # print frequency of each class
        print(np.unique(prediction, return_counts=True))
        return outs


if __name__ == '__main__':
    try:
        # read pickle model
        nn = pickle.load(open('nn-25-sigm.pkl', 'rb'))
        # nn = pickle.load(open('nn-100-relu.pkl', 'rb'))
        print("model loaded")
    except:
        activation_type = 'relu'
        train_data = pn.get_data(
            'poker_dataset\poker-hand-training-true-onehot.data')

        features_num = len(train_data.iloc[0])-1
        hidden_size = [100, 100]
        hidden_number = len(hidden_size)

        target_num = 10
        epochs = 300
        batch_size = 100
        learn_rate = 0.1

        nn = NeuralNetwork(features_num=features_num, hidden_number=hidden_number, hidden_size=hidden_size,
                           target_num=target_num, activation_type=activation_type, m=batch_size, adaptive=True)

        nn.train(train_data.iloc[:, :-1].T,
                 train_data.iloc[:, -1], learn_rate, epochs, batch_size)
        # pickle.dump(nn, open('nn-100-relu.pkl', 'wb'))
        pickle.dump(nn, open('nn-25-sigm.pkl', 'wb'))
        print("model saved")
    # nn = pickle.load(open('nn-100-relu.pkl', 'rb'))
    nn = pickle.load(open('nn-25-sigm.pkl', 'rb'))
    test_data = pn.get_data('poker_dataset\poker-hand-testing-onehot.data')
    nn.test(test_data.iloc[:, :-1].T, test_data.iloc[:, -1])
    test_data = None
    test_data = pn.get_data(
        'poker_dataset\poker-hand-training-true-onehot.data')
    nn.test(test_data.iloc[:, :-1].T, test_data.iloc[:, -1])
