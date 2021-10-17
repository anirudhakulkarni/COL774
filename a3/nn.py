import numpy as np
import preprocess_nn as pn

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class perceptrons():
    '''
    wT*x+b = y
    perceptron class is collection of perceptrons in the layer
    arguments:
        input_feature_size: size of input vector, i.e. number of perceptrons in the previous layer
        perceptron_count: number of perceptrons in the layer
        type: layer type, i.e. input, hidden, output
    '''

    def __init__(self, input_feature_size, perceptron_count, type):
        self.weights = np.zeros((input_feature_size, perceptron_count))
        self.biases = np.zeros((perceptron_count, 1))
        self.type = type
        self.perceptron_count = perceptron_count
        self.input_feature_size = input_feature_size
        self.delta = np.zeros((perceptron_count, 1))
        self.output = np.zeros((perceptron_count, 1))

    def calculate_delta(self, target, perceptrons_next, type):
        if type == 'output':

            self.delta = np.sum(np.multiply(np.multiply(target - self.output,self.output), (1 - self.output)),axis=1,keepdims=True)
        elif type == 'hidden':
            print(perceptrons_next.weights)
            print(perceptrons_next.delta)
            print(self.output)
            self.delta = np.sum(np.multiply(np.multiply(np.dot(perceptrons_next.weights, perceptrons_next.delta), self.output), (1 - self.output)),axis=1,keepdims=True)
            print(self.delta)
            # self.delta = np.dot(perceptrons_next.weights,
                                # perceptrons_next.delta) * self.output * (1 - self.output)
        return self.delta

    def update_weights(self, input_vector, learning_rate):
        print("updating...")
        print(self.weights)
        print(input_vector)
        print(self.delta)
        self.weights = self.weights + learning_rate * self.delta.T
            # np.dot(input_vector.T, self.delta)
        self.biases = self.biases + learning_rate * self.delta
    def predict(self, input_vector):
        print(self.weights.shape)
        print(input_vector.shape)
        print(self.biases.shape)
        self.output = sigmoid(np.dot(self.weights.T, input_vector) + self.biases)
        return self.output


class NeuralNetwork():
    '''
    Assumptions:
        1. input_vector is (features_num, example_num)
        2. label is (target_num, example_num)

    '''
    def __init__(self, features_num, hidden_number, hidden_size, target_num):
        self.features_num = features_num
        self.hidden_number = hidden_number
        self.hidden_size = hidden_size
        self.target_num = target_num
        self.input_layer = perceptrons(features_num, features_num, "input")
        self.hidden_layers = [perceptrons(
            features_num, hidden_size[0], "hidden")]
        self.hidden_layers += [perceptrons(hidden_size[i-1], hidden_size[i], "hidden")
                               for i in range(1, hidden_number)]
        self.output_layer = perceptrons(hidden_size[-1], target_num, "output")

    def train(self, input_vector, label, learn_rate,epochs, batch_size):
        for i in range(epochs):
            self.predict(input_vector)
            # print(self.output_layer)
            self.output_layer.calculate_delta(label, self.output_layer, "output")
            print(self.output_layer.delta)

            for h in range(self.hidden_number-1,-1,-1):
                if h == self.hidden_number-1:
                    self.hidden_layers[h].calculate_delta(
                        label, self.output_layer, "hidden")
                else:
                    self.hidden_layers[h].calculate_delta(label, self.hidden_layers[h+1], "hidden")
                self.hidden_layers[h].update_weights(input_vector, learn_rate)
            self.input_layer.update_weights(input_vector, learn_rate)

    def predict(self, input_vector):
        last_out = self.input_layer.predict(input_vector)
        for h in self.hidden_layers:
            last_out = h.predict(last_out)
        return self.output_layer.predict(last_out)

    def test(self, input_vector, label):
        prediction = self.predict(input_vector)
        return np.argmax(prediction) == np.argmax(label)


# main 
if __name__ == "__main__":
    # train
    features_num = 6000
    hidden_number = 1
    hidden_size = [100]
    target_num = 5
    epochs = 100
    batch_size = 1
    learn_rate = 0.1
    nn = NeuralNetwork(features_num, hidden_number, hidden_size, target_num)
    # print initialization
    print(nn.input_layer.weights)
    print(nn.input_layer.biases)
    for h in nn.hidden_layers:
        print(h.weights)
        print(h.biases)
    print(nn.output_layer.weights)
    print(nn.output_layer.biases)
    train_data=pn.get_data('poker_dataset/temp.data')
    # print(train_data[:,:-1])
    nn.train(train_data.iloc[:,:-1], train_data.iloc[:,-1], 0.1, 100, len(train_data))
