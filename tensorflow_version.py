import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

training_data_X = [[0,1], [0,0], [1,0], [1,1]]
training_data_y = [[1], [0], [1], [0]]

model = Sequential()
model.add(Dense(units=2, activation='sigmoid'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(training_data_X, training_data_y, epochs=500, batch_size=4)