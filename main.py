import numpy as np

def relu(x) -> float:
    return np.maximum(0, x)

def derivative_relu(x) -> float:
    if np.maximum(0, x) == x:
        return 1
    
    return 0

class Neuron:
    def __init__(self, num_connections: int):
        self.weights = np.random.rand(num_connections)
        self.bias = np.zeros(1)
        self.delta = 1.0 # This is used for calculating the error
        self.output = 1.0 # This is also used for calculating the error

    def activate(self, x: np.matrix) -> np.matrix:
        y = np.dot(x, self.weights)

        # TODO: Bias sum

        y = relu(y)
        self.output = y # Storing last output value for backpropagation
        return y

class NeuralNetwork:
    def __init__(self, num_layers: int, width: int, input_size: int, output_size: int):
        self.num_layers = num_layers
        self.width = width
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.build()

    def build(self):
        neurons_to_create = self.num_layers * self.width
        num_previous_nodes = self.input_size

        # Creating now the hidden layers
        while neurons_to_create > 0:
            layer = []

            for _ in range(self.width):
                layer.append(Neuron(num_connections=num_previous_nodes))
            
            self.layers.append(layer)
            neurons_to_create = neurons_to_create - self.width
            num_previous_nodes = self.width
        
        # Creating now the output layer
        output_layer = []
        for _ in range(self.output_size):
            output_layer.append(Neuron(num_connections=num_previous_nodes))
        
        self.layers.append(output_layer)

    def predict(self, x) -> list:
        Z_predicted = x

        # Calculating with Hidden Layers
        for layer_index in range(self.num_layers):
            for width_index in range(self.width):  
                neuron = self.layers[layer_index][width_index]
                Z = neuron.activate(Z_predicted)
                Z_predicted[width_index] = Z

        # Calculating with Output Layer
        result = []
        for output_node_index in range(self.output_size):
            neuron = self.layers[-1][output_node_index]
            Z = neuron.activate(Z_predicted)
            result.append(Z)

        return result

    def backpropagate(self, y_real, y_predicted):
        output_layer_index = len(self.layers) - 1
        errors = []

        for layer_index in range(len(self.layers)):
            # We are on the output layer
            if layer_index == output_layer_index:
                for neuron_index in range(len(y_real)):
                    neuron = self.layers[layer_index][neuron_index]
                    errors.append(y_predicted[neuron_index] - y_real[neuron_index])
            else: # We are on the hidden layer
                for neuron_index in range(self.width):
                    error = 0.0
                    neuron = self.layers[layer_index][neuron_index]
                    for weight_index in range(len(neuron.weights)):
                        error += neuron.weights[weight_index] * neuron.delta
                    errors.append(error)

            amount_neurons = self.width if layer_index != output_layer_index else len(y_real)
            for neuron_index in range(amount_neurons):
                neuron = self.layers[layer_index][neuron_index]
                neuron.delta = errors[neuron_index] * derivative_relu(neuron.output)

    def update_weights(self, training_data_X, learning_rate = 0.001):
        pass

    def train(self, num_epochs: int, training_data_X, training_data_y, learning_rate = 0.001):
        for epoch in range(num_epochs):

            # Variables
            error = 0.0
            amount_of_correct = 0

            for index in range(len(training_data_X)):
                Z = self.predict(training_data_X[index])
                self.backpropagate(training_data_y[index], Z)
                self.update_weights(training_data_X=training_data_X[index], learning_rate=learning_rate)
                print("[TRAINING][EPOCH {0}] Error: {1}".format(epoch + 1, self.layers[-1][0].delta))




if __name__ == '__main__':
    #training_data_X = [[0,0], [0,1], [1,0], [1,1]]
    #training_data_y = [[0], [1], [1], [0]]

    training_data_X = [[0,1], [1,0], [1,1]]
    training_data_y = [[1], [1], [0]]

    neural_network = NeuralNetwork(num_layers=1, width=2, input_size=2, output_size=1)
    neural_network.train(num_epochs=100, training_data_X=training_data_X, training_data_y=training_data_y, learning_rate=0.1)
