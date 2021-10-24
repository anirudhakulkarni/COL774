import numpy as np
from numpy.core.shape_base import hstack
from numpy.lib.scimath import sqrt
import preprocess_nn as pn
np.random.seed(10)


def activation(a, type):
    if type == 'sigmoid':
        # a1 = np.multiply(a >= 0, a)
        # a2 = np.multiply(a < 0, a)
        # return np.add(1/(1+np.exp(-a1)), np.divide(np.exp(a2), (1+np.exp(a2)))) - 0.5
        return 1 / (1 + np.exp(-a))
    elif type == 'tanh':
        return np.tanh(x)
    elif type == 'relu':
        return np.maximum(0, x)


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
        self.weights = np.zeros((input_feature_size, perceptron_count))
        # create random weights of size input_feature_size x perceptron_count
        # self.weights = np.random.uniform(-1/sqrt(input_feature_size), 1/sqrt(input_feature_size),
        #                                  (input_feature_size, perceptron_count))
        self.type = type
        self.perceptron_count = perceptron_count
        self.input_feature_size = input_feature_size
        self.gradients = np.zeros((input_feature_size, perceptron_count))
        self.output = None
        self.activation_type = activation_type
        self.netj = None
        self.input = None
        self.m = 100
        self.b = np.zeros((perceptron_count, 1))

    def calculate_delta(self,  perceptrons_next=None, target=None):
        if self.type == 'output':
            # print("calculating delta for output layer")
            # print("target shape", target.shape)
            # print("output shape", self.output.shape)
            # print(self.output)
            # print(target)
            self.delta = np.multiply(np.multiply(
                target - self.output, self.output), (1 - self.output))
            self.gradients = - np.dot(self.input, self.delta.T)/self.m
        elif self.type == 'hidden':
            self.delta = np.multiply(np.multiply(np.dot(
                perceptrons_next.weights, perceptrons_next.delta), self.output), (1 - self.output))
            self.gradients = - np.dot(self.input, self.delta.T)/self.m
            # self.delta = np.dot(perceptrons_next.weights,
            # perceptrons_next.delta) * self.output * (1 - self.output)
        return self.gradients

    def update_weights(self,  learning_rate):
        # print("updating...")
        self.weights = self.weights - learning_rate * self.gradients
        # print("PRinting bbbbbbbbbbbbbbbbbbbbbbbbb")
        # print(self.b)
        # print(self.delta)
        # print((np.sum(self.delta, axis=1)).reshape(-1, 1))
        self.b = self.b + learning_rate * \
            (np.sum(self.delta, axis=1)).reshape(-1, 1)/self.m
        # print("Shapes after updating:")
        # print(self.weights.shape)
        # print(self.b.shape)
        # np.dot(input_vector.T, self.delta)

    def predict(self, input_vector):
        self.netj = np.dot(self.weights.T, input_vector)
        self.netj = self.netj + self.b
        self.output = activation(self.netj, self.activation_type)
        # print("Output shape:", self.output.shape)
        # if self.type == 'output':
        # print("HERE")
        # print(self.weights)
        return self.output


class NeuralNetwork():
    '''
    Assumptions:
        1. input_vector is (features_num, example_num)
        2. label is (target_num, example_num)

    '''

    def __init__(self, features_num, hidden_number, hidden_size, target_num, activation_type, m):
        self.features_num = features_num
        self.hidden_number = hidden_number
        self.hidden_size = hidden_size
        self.target_num = target_num
        self.input_layer = perceptrons(
            features_num, features_num, "input", activation_type, m)
        self.hidden_layers = [perceptrons(
            features_num, hidden_size[0], "hidden", activation_type, m)]
        self.hidden_layers += [perceptrons(hidden_size[i-1], hidden_size[i], "hidden", activation_type, m)
                               for i in range(1, hidden_number)]
        self.output_layer = perceptrons(
            hidden_size[-1], target_num, "output", 'sigmoid', m)
        self.activation_type = activation_type

    def train(self, input_vector, label, learn_rate, epochs, batch_size):
        # create sparse matrix with corresponding label as 1
        label_mat = np.zeros((self.target_num, label.shape[0]))
        for i in range(label.shape[0]):
            label_mat[int(label[i])][i] = 1
        label_mat = np.matrix(label_mat)
        # print("Label")
        # print(label_mat)
        # print(label_mat.shape)
        label = label_mat
        self.input_layer.input = input_vector
        for i in range(epochs):
            print("epoch: ", i)
            for j in range(0, input_vector.shape[1], batch_size):
                # print("batch: ", j)
                self.input_layer.input = input_vector.iloc[:, j:j+batch_size]
                label_mat_batch = label[:, j:j+batch_size]
                input_vector_batch = input_vector.iloc[:, j:j+batch_size]
# zaviar initialization
                self.forward_pass(input_vector_batch)
                # print(self.output_layer)
                self.output_layer.calculate_delta(target=label_mat_batch)
                for h in range(self.hidden_number-1, -1, -1):
                    if h == self.hidden_number-1:
                        self.hidden_layers[h].calculate_delta(
                            self.output_layer)
                    else:
                        self.hidden_layers[h].calculate_delta(
                            self.hidden_layers[h+1])
                self.output_layer.update_weights(learn_rate)
                for h in range(self.hidden_number-1, -1, -1):
                    self.hidden_layers[h].update_weights(learn_rate)
                # print("Printing delta")
                # print(self.hidden_layers[0].delta)
                # print(self.output_layer.delta)
                # print("Printing weights")
                # print(self.hidden_layers[-1].weights)
                # print(self.output_layer.weights)
                # print("Printing b")
                # print(self.hidden_layers[-1].b)
                # print(self.output_layer.b)
                # print error
                # self.forward_pass(input_vector_batch)
                # error = np.sum(
                #     np.square(label_mat_batch - self.output_layer.output))/input_vector_batch.shape[1]
                # print(error)
            self.forward_pass(input_vector)
            error = np.sum(
                np.square(label_mat - self.output_layer.output))/input_vector.shape[1]
            print(error)
            # self.input_layer.update_weights(input_vector, learn_rate)

    def backprop(self, label):
        '''label is (6000 X 1)'''

        self.output_layer.calculate_delta(target=label)
        self.output_layer.update_weights()
        for h in range(self.hidden_number-1, -1, -1):
            if h == self.hidden_number-1:
                self.hidden_layers[h].calculate_delta(
                    perceptrons_next=self.output_layer)
            else:
                self.hidden_layers[h].calculate_delta(
                    perceptrons_next=self.hidden_layers[h+1])
            self.hidden_layers[h].update_weights()

    def forward_pass(self, input_vector):
        # last_out = self.input_layer.predict(input_vector)
        last_out = None
        for h in self.hidden_layers:
            if last_out is None:
                # append row of ones to input_vector
                # input_vector = np.vstack(
                #     (np.ones((1, input_vector.shape[1])), input_vector))

                h.input = input_vector
                last_out = h.predict(input_vector)
            else:
                # last_out = np.hstack(
                #     (np.ones((1, last_out.shape[1])), last_out))
                h.input = last_out

                last_out = h.predict(last_out)
            # print(last_out.shape)

        # print(last_out.shape)
        # last_out = np.vstack((np.ones((1, last_out.shape[1])), last_out))
        # print(last_out.shape)
        self.output_layer.input = last_out
        return self.output_layer.predict(last_out)

    def test(self, input_vector, label):
        prediction = self.forward_pass(input_vector)
        # print(prediction)
        t1 = np.argmax(prediction, axis=0)
        print(t1)
        t2 = np.zeros(label.shape[0])
        for i in range(t1.shape[0]):
            t2[i] = t1.item((0, i))
        print(prediction)
        # print(prediction.shape)
        # print occurences of each class
        # print(t2)
        # save array as csv
        np.savetxt('prediction.csv', t2)
        # read csv
        t2 = np.array(np.genfromtxt('prediction.csv', delimiter=','))
        # print(t2)
        # print(t2.shape)
        # print(label)
        # print(label.shape)
        print(np.unique(t2, return_counts=True))
        print(np.unique(label, return_counts=True))
        print(label)

        outs = np.count_nonzero(t2 == label)
        print(outs/label.shape[0])
        return outs


# main
if __name__ == "__main__":
    # train
    train_data = pn.get_data(
        'poker_dataset\poker-hand-training-true-onehot.data')
    # train_data = pn.get_data(
    #     'temp.csv')
    features_num = len(train_data.iloc[0])-1
    hidden_number = 1
    hidden_size = [25]
    target_num = 10
    epochs = 100
    batch_size = 100
    learn_rate = 0.1
    nn = NeuralNetwork(features_num, hidden_number,
                       hidden_size, target_num, 'sigmoid', len(train_data))
    # print initialization
    nn.train(train_data.iloc[:, :-1].T,
             train_data.iloc[:, -1], learn_rate, epochs, batch_size)
    print(nn.test(train_data.iloc[:, :-1].T, train_data.iloc[:, -1]))
    try:
        lastweights = np.loadtxt('weights.csv', delimiter=',')
        # compare if weights are same
        # print(lastweights.shape)
        # print(nn.hidden_layers[0].weights.shape)
        if np.array_equal(lastweights, nn.hidden_layers[0].weights):
            print("weights are same")
        else:
            print("weights are different")
        np.savetxt(
            'weights.csv', nn.hidden_layers[0].weights, delimiter=',')

    except:
        np.savetxt(
            'weights.csv', nn.hidden_layers[0].weights, delimiter=',')
        print("filenotfound")


''''
neuron 1        neuron 2
theta0          theta0
theta1          theta1
theta2          theta2
.
.
.
theta(nfeature-1)

'''
'''

'''
