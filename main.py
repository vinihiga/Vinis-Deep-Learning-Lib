import numpy as np
import math

def sigmoid(x) -> float:
    return 1 / 1 + math.exp(-x)

def derivative_sigmoid(x) -> float:
    return sigmoid(x)*(1 - sigmoid(x))

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
        self.input = 1.0 # This is also used for calculating the error

    def activate(self, x: np.matrix) -> np.matrix:
        y = np.dot(x, self.weights)

        # TODO: Bias sum

        y = sigmoid(y)
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
        result = []
        current_x = x
        debug_inputs = [] # DEBUG

        # Calculating with Hidden Layers
        for layer_index in range(self.num_layers):
            debug_inputs_current_layer = [] # DEBUG
            for width_index in range(self.width):  
                neuron = self.layers[layer_index][width_index]
                neuron.input = current_x
                Z = neuron.activate(current_x)
                neuron.output = Z
                current_x[width_index] = Z
                debug_inputs_current_layer.append(current_x) # DEBUG
            
            debug_inputs.append(debug_inputs_current_layer)

        # Calculating with Output Layer
        debug_inputs_output_layer = [] # DEBUG
        for output_node_index in range(self.output_size):
            neuron = self.layers[-1][output_node_index]
            neuron.input = current_x
            Z = neuron.activate(current_x)
            neuron.output = Z
            result.append(Z)
            debug_inputs_output_layer.append(current_x) # DEBUG

        debug_inputs.append(debug_inputs_output_layer)

        #print("Current inputs: {0}".format(debug_inputs))

        return result

    def backpropagate(self, y_real, y_predicted):
        output_layer_index = len(self.layers) - 1
        errors = []

        for layer_index in range(len(self.layers)):
            if layer_index == output_layer_index: # We are on the output layer
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
                neuron.delta = errors[neuron_index] * derivative_sigmoid(neuron.output)

    def update_weights(self, learning_rate = 0.001):
        for layer_index in range(len(self.layers)):
            for neuron in self.layers[layer_index]:
                if layer_index == len(self.layers) - 1:
                    neuron.weights[index] -= learning_rate * neuron.delta
                else:
                    for index in range(len(neuron.input)):
                        neuron.weights[index] -= learning_rate * neuron.input[index] * neuron.delta


    def train(self, num_epochs: int, training_data_X, training_data_y, learning_rate = 0.001):
        for epoch in range(num_epochs):

            # Variables
            amount_of_correct = 0
            training_data_X_len = len(training_data_X)
            avg_error = 0.0

            for index in range(training_data_X_len):
                Z = self.predict(training_data_X[index])
                self.backpropagate(training_data_y[index], Z)
                self.update_weights(learning_rate=learning_rate)

                for neuron in self.layers[-1]:
                    avg_error += neuron.delta

                    if index != 0:
                        avg_error = avg_error / 2.0

                for Z_index in range(len(Z)):
                    Z[Z_index] = 1 if Z[Z_index] >= 0.5 else 0

                if Z == training_data_y[index]:
                    amount_of_correct += 1

            print("[TRAINING][EPOCH {0}] Error: {1}    Accuracy: {2}".format(epoch + 1, avg_error, amount_of_correct / training_data_X_len))

        print("[TRAINING FINISHED]\n\n\n")




if __name__ == '__main__':
    training_data_X = [[0,0], [0,1], [1,0], [1,1]]
    training_data_y = [[0], [1], [1], [0]]

    neural_network = NeuralNetwork(num_layers=1, width=2, input_size=2, output_size=1)
    neural_network.train(num_epochs=3, training_data_X=training_data_X, training_data_y=training_data_y, learning_rate=0.001)

    Z = neural_network.predict([1, 1])
    Z[0] = 1 if Z[0] >= 0.5 else 0

    print("[PREDICT] Result: {0}".format(Z))
