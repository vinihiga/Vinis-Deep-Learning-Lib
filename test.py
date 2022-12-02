from sklearn import datasets
from sklearn.model_selection import train_test_split
from vini import NeuralNetwork, Neuron, sigmoid, derivative_sigmoid, relu, derivative_relu

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y_raw = iris.target
y = []

for y_real in y_raw:
    if y_real == 0:
        y.append([1, 0, 0])
    elif y_real == 1:
        y.append([0, 1, 0])
    else:
        y.append([0, 0, 1])

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

neural_network = NeuralNetwork(input_size=len(x_train[0]), output_size=len(y_train[0]))
neural_network.add_layer(width=512, activation_function=relu, derivative_function=derivative_relu)
neural_network.add_layer(width=10, activation_function=relu, derivative_function=derivative_relu)
neural_network.train(num_epochs=100, training_data_X=X, training_data_y=y, learning_rate=0.001, verbose=True)

# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import pandas as pd

# model = Sequential()
# model.add(Dense(units=512, activation='sigmoid'))
# model.add(Dense(units=10, activation='sigmoid'))
# model.add(Dense(units=3, activation='softmax'))
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# y_train = pd.DataFrame(y_train, columns = ['cat_1', 'cat_2', 'cat_3'])
# y_test = pd.DataFrame(y_test, columns = ['cat_1', 'cat_2', 'cat_3'])

# model.fit(x_train, y_train, epochs=100, batch_size=10)
# model.evaluate(x_test, y_test)