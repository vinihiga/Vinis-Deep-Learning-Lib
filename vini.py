import numpy as np
import math

def sigmoid(x) -> float:
    return 1 / (1 + math.exp(-x))

def derivative_sigmoid(x) -> float:
    return sigmoid(x)*(1 - sigmoid(x))

def relu(x) -> float:
    return np.maximum(0, x)

def derivative_relu(x) -> float:
    if x > 0:
        return 1
    
    return 0

class Neuron:
    def __init__(self, num_connections: int, activation_function, derivative_function):
        self.num_connections = num_connections
        self.activation_function = activation_function
        self.derivative_function = derivative_function

        self.delta = 1.0 # This is used for calculating the error
        self.output = 1.0 # This is also used for calculating the error
        self.input = 1.0 # This is also used for calculating the error

        self.weights = self.__weight_initialization__()
        self.bias = np.zeros(1)

    def activate(self, x: np.matrix) -> np.matrix:
        y = np.dot(x, self.weights)
        #y = np.sum(y, self.bias)
        z = self.activation_function(y)
        self.output = z # Storing last output value for backpropagation
        return z

    def __weight_initialization__(self):
        return np.random.randn(self.num_connections)

class NeuralNetwork:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []

    def add_layer(self, width: int, activation_function, derivative_function):
        num_previous_nodes = self.input_size
        neurons_to_create = width

        if len(self.layers) != 0:
            num_previous_nodes = len(self.layers[-2])

        # Creating the hidden layer
        layer = []

        for _ in range(neurons_to_create):
            layer.append(Neuron(num_connections=num_previous_nodes, activation_function=activation_function, derivative_function=derivative_function))
        
        if len(self.layers) == 0:
            self.layers.append(layer)
        else:
            self.layers[-1] = layer

        # Creating the output layer
        num_previous_nodes = width
        neurons_to_create = self.output_size
        layer = []

        for _ in range(neurons_to_create):
            layer.append(Neuron(num_connections=num_previous_nodes, activation_function=activation_function, derivative_function=derivative_function))
        
        self.layers.append(layer)

    def predict(self, x: list) -> list:

        if len(self.layers) == 0:
            raise Exception("Please add the layers before predicting.")

        result = []
        current_x = x

        # Calculating with Hidden Layers
        for layer_index in range(len(self.layers) - 1):
            current_layer_size = len(self.layers[layer_index])
            new_x = np.zeros(current_layer_size)

            for width_index in range(current_layer_size):  
                neuron = self.layers[layer_index][width_index]
                neuron.input = current_x
                Z = neuron.activate(current_x)
                neuron.output = Z
                new_x[width_index] = Z
            
            current_x = new_x

        # Calculating with Output Layer
        for output_node_index in range(self.output_size):
            neuron = self.layers[-1][output_node_index]
            neuron.input = current_x
            Z = neuron.activate(current_x)
            neuron.output = Z
            result.append(Z)

        #print("PREDICTED: " + str(result[0]))

        return result

    def backpropagate(self, y_real, y_predicted, learning_rate = 0.001):
        for layer_index in reversed(range(len(self.layers))):
            errors = []
            
            # Imagining the following Neural Network
            # 
            # Input_1 -> H_1 -> Out_1
            # Input_2 ----/\

            # The weight are W1 and W2 for inputs -> H_1
            # And the last weight W3 is H_1 -> Out_1

            if layer_index == len(self.layers) - 1: # We are on the output layer
                for neuron_index in range(len(y_real)):
                    neuron = self.layers[layer_index][neuron_index]
                    # dE/dW2 = dE/dO * dO/dZ * dZ/dW2
                    derivative_output = -(y_real[neuron_index] - y_predicted[neuron_index])
                    derivative_activation = neuron.derivative_function(neuron.output)
                    derivative_linear_function = np.sum(neuron.input)
                    errors.append(derivative_output * derivative_activation * derivative_linear_function)
            else: # We are on the hidden layer
                for neuron_index in range(len(self.layers[layer_index])):
                    neuron = self.layers[layer_index][neuron_index]

                    # dE / dW1 = dE / dO * dO/dZ * dZ/dW1
                    derivative_output = 0

                    # Getting the deltas from parents
                    for parent_index in range(len(self.layers[layer_index + 1])):
                        derivative_output += self.layers[layer_index + 1][parent_index].delta

                    derivative_activation = neuron.derivative_function(neuron.output)
                    derivative_linear_function = np.sum(neuron.input)

                    errors.append(derivative_output * derivative_activation * derivative_linear_function)

            amount_neurons = len(self.layers[layer_index]) if layer_index != (len(self.layers) - 1) else len(y_real)
            for neuron_index in range(amount_neurons):
                neuron = self.layers[layer_index][neuron_index]
                neuron.delta = errors[neuron_index] #* neuron.derivative_function(neuron.output) #derivative_relu(neuron.output)
                
                for weight_index in range(len(neuron.weights)):
                    neuron.weights[weight_index] -= learning_rate * errors[neuron_index]

    def train(self, num_epochs: int, training_data_X, training_data_y, learning_rate = 0.001, verbose = True):
        for epoch in range(num_epochs):

            # Variables
            amount_of_correct = 0
            training_data_X_len = len(training_data_X)
            avg_error = 0.0

            for data_index in range(training_data_X_len):
                Z = self.predict(training_data_X[data_index])
                self.backpropagate(training_data_y[data_index], Z, learning_rate)

                # TODO: Check if loss (MSE) is correct for N outputs for each training data
                for output_index in range(len(Z)):
                    abs_error = Z[output_index] - training_data_y[data_index][output_index]
                    avg_error += pow(abs_error, 2) / 2.0

                # Accuracy - for debugging purposes
                for Z_index in range(len(Z)):
                    Z[Z_index] = 1 if Z[Z_index] >= 0.5 else 0

                if Z == training_data_y[data_index]:
                    amount_of_correct += 1

            if verbose == True and (epoch + 1) % 10 == 0:
                print("[TRAINING][EPOCH {0}] Error: {1}    Accuracy: {2}".format(epoch + 1, avg_error, amount_of_correct / training_data_X_len))

if __name__ == '__main__':

    # This is used for testing XOR

    print("[ STARTING XOR TEST ]")

    training_data_X = [[1,0], [0,0], [1,1], [0,1]]
    training_data_y = [[1], [0], [0], [1]]

    neural_network = NeuralNetwork(input_size=len(training_data_X[0]), output_size=len(training_data_y[0]))
    neural_network.add_layer(width=2, activation_function=sigmoid, derivative_function=derivative_sigmoid)
    neural_network.add_layer(width=1, activation_function=sigmoid, derivative_function=derivative_sigmoid)
    neural_network.train(num_epochs=50000, training_data_X=training_data_X, training_data_y=training_data_y, learning_rate=0.001, verbose=True)

    # Z = neural_network.predict([1, 1])
    # Z[0] = 1 if Z[0] >= 0.5 else 0

    # print("[PREDICT] Result: {0}".format(Z))
