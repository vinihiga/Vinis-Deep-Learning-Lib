import numpy as np
import math
import json
import random

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

        self.delta = [0.0] * num_connections # This is used for calculating the error
        self.output = 0.0 # This is also used for calculating the error
        self.input = 0.0 # This is also used for calculating the error

        self.weights = self.__weight_initialization__(should_use_he_init = False)
        self.bias = np.zeros(1)

    def activate(self, x: np.matrix) -> np.matrix:
        y = np.dot(x, self.weights)
        y += self.bias
        z = self.activation_function(y)
        self.output = z # Storing last output value for backpropagation
        return z

    def __weight_initialization__(self, should_use_he_init: bool):
        weight = np.random.randn(self.num_connections)

        # Using He initialization
        if should_use_he_init:
            weight = weight * math.sqrt(2 / self.num_connections)

        return weight

class NeuralNetwork:
    def __init__(self, input_size: int):
        self.input_size = input_size
        self.layers = []

    def print_num_of_connections(self, layer_index: int):
        print("[STATUS] Num. of connections in layer {0}: {1}".format(layer_index, self.layers[layer_index][0].num_connections))

    def add_layer(self, width: int, activation_function: str):
        num_previous_nodes = self.input_size
        neurons_to_create = width

        if len(self.layers) != 0:
            num_previous_nodes = len(self.layers[-1])

        # Creating the hidden layer
        layer = []

        for _ in range(neurons_to_create):
            activation_ref = None
            derivative_ref = None

            if activation_function == "sigmoid":
                activation_ref = sigmoid
                derivative_ref = derivative_sigmoid
            elif activation_function == "relu":
                activation_ref = relu
                derivative_ref = derivative_relu
            elif activation_ref == None or derivative_ref == None:
                raise Exception("No activation or derivative functions was passed")

            layer.append(Neuron(num_connections=num_previous_nodes, activation_function=activation_ref, derivative_function=derivative_ref))
        
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
        for output_node_index in range(len(self.layers[-1])):
            neuron = self.layers[-1][output_node_index]
            neuron.input = current_x
            Z = neuron.activate(current_x)
            neuron.output = Z
            result.append(Z)

        #print("PREDICTED: " + str(result[0]))

        return result

    def backpropagate(self, y_real, y_predicted, learning_rate = 0.001):
        for layer_index in reversed(range(len(self.layers))):

            if layer_index == len(self.layers) - 1: # We are on the output layer
                for neuron_index in range(len(y_real)):
                    neuron = self.layers[layer_index][neuron_index]

                    for weight_index in range(len(neuron.weights)):
                        # dE/dW2 = dE/dO * dO/dZ * dZ/dW2
                        previous_weight = neuron.weights[weight_index]

                        derivative_error = -(y_real[neuron_index] - y_predicted[neuron_index])
                        derivative_activation = neuron.derivative_function(neuron.output)
                        derivative_linear_function = neuron.input[weight_index]
                        delta = derivative_error * derivative_activation * derivative_linear_function
                        neuron.delta[weight_index] = derivative_error * derivative_activation # Output layer is different from hidden layer
                        neuron.weights[weight_index] = neuron.weights[weight_index] - (learning_rate * delta)

                        #print("[STATUS] Updated weight {0},{1} from {2} to {3}".format(layer_index, weight_index, previous_weight, neuron.weights[weight_index]))

            else: # We are on the hidden layer
                for neuron_index in range(len(self.layers[layer_index])):
                    neuron = self.layers[layer_index][neuron_index]

                    for weight_index in range(len(neuron.weights)):
                        # dE / dW1 = previous(dE / dO * dO/dZ) * current(dO/dZ * dZ/dW1)
                        previous_weight = neuron.weights[weight_index]
                        derivative_errors = 0

                        # Getting the deltas from parents
                        for parent_index in range(len(self.layers[layer_index + 1])):
                            derivative_errors += self.layers[layer_index + 1][parent_index].delta[neuron_index]
                            #print("[STATUS] Current derivative errors from parent {0},{1} : {2}".format(layer_index + 1, parent_index, derivative_errors))


                        derivative_activation = neuron.derivative_function(neuron.output)
                        derivative_linear_function = neuron.input[weight_index]
                        delta = derivative_errors * derivative_activation * derivative_linear_function
                        neuron.delta[weight_index] = delta
                        neuron.weights[weight_index] = neuron.weights[weight_index] - (learning_rate * delta)

                        #print("[STATUS] Updated weight {0},{1} from {2} to {3}".format(layer_index, weight_index, previous_weight, neuron.weights[weight_index]))


    def save(self):
        data = {
            "layers": []
        }

        for layer_index in range(len(self.layers)):
            layer = []

            for neuron_index in range(len(self.layers[layer_index])):
                neuron = self.layers[layer_index][neuron_index]
                weights = []
                bias = []

                if type(neuron.weights) == list:
                    weights = neuron.weights
                else:
                    weights = neuron.weights.tolist()

                if type(neuron.bias) == list:
                    bias = neuron.bias
                else:
                    bias = neuron.bias.tolist()

                if len(weights) == 0 or len(bias) == 0:
                    raise Exception("Fatal error! Couldn't resolve weights or bias!")

                actual_neuron_data = {
                    "activation_name": neuron.activation_function.__name__,
                    "weights": weights,
                    "bias": bias
                }

                layer.append(actual_neuron_data)
            
            data["layers"].append(layer)

        print("[SAVING DATA]")
        with open('checkpoint.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def load(path: str, input_size: int) -> 'NeuralNetwork':
        neural_network = NeuralNetwork(input_size=input_size)
        print("[LOADING DATA]")

        with open('checkpoint.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

            for layer_index in range(len(data["layers"])):
                width = len(data["layers"][layer_index])
                activation_name = data["layers"][layer_index][0]["activation_name"]
                neural_network.add_layer(width, activation_name)
                
                for neuron_index in range(len(data["layers"][layer_index])):
                    neural_network.layers[-1][neuron_index].weights = data["layers"][layer_index][neuron_index]["weights"]
                    neural_network.layers[-1][neuron_index].bias = data["layers"][layer_index][neuron_index]["bias"]
        
        return neural_network

    def train(self, num_epochs: int, training_data_X, training_data_y, learning_rate = 0.001, verbose = True):
        for epoch in range(num_epochs):

            # Variables
            amount_of_correct = 0
            training_data_X_len = len(training_data_X)
            avg_error = 0.0

            for data_index in range(training_data_X_len):
                Z = self.predict(training_data_X[data_index])
                self.backpropagate(training_data_y[data_index], Z, learning_rate)

                # TODO: Change loss functions... Allow the user to select
                for output_index in range(len(Z)):
                    abs_error = training_data_y[data_index][output_index] - Z[output_index]
                    avg_error += pow(abs_error, 2)

                # Accuracy - for debugging purposes
                for Z_index in range(len(Z)):
                    Z[Z_index] = 1 if Z[Z_index] >= 0.5 else 0

                if Z == training_data_y[data_index]:
                    amount_of_correct += 1

            if verbose == True and (epoch + 1) % 100 == 0:
                avg_error = avg_error / training_data_X_len
                print("[TRAINING][EPOCH {0}] Error: {1}    Accuracy: {2}".format(epoch + 1, avg_error, amount_of_correct / training_data_X_len))

if __name__ == '__main__':

    # This is used for testing XOR

    print("[ STARTING XOR TEST ]")

    training_data_X = [[1,0], [0,0], [1,1], [0,1]]
    training_data_y = [[1], [0], [0], [1]]

    neural_network = NeuralNetwork(input_size=len(training_data_X[0]))
    neural_network.add_layer(width=2, activation_function="relu")
    neural_network.add_layer(width=len(training_data_y[0]), activation_function="sigmoid")
    neural_network.train(num_epochs=50000, training_data_X=training_data_X, training_data_y=training_data_y, learning_rate=0.3, verbose=True)
    #neural_network.save()

    #neural_network = NeuralNetwork.load("./checkpoint.json", input_size=len(training_data_X[0]))
    #neural_network.save()

    Z = neural_network.predict([0, 0])
    Z[0] = 1 if Z[0] >= 0.5 else 0

    print("[PREDICT] Result: {0}".format(Z))
