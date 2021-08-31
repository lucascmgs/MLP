import mlb
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
import mlb

def normalization(data):
  return data.astype('float32') / 255

def to_categorical(data):
  new_data = np.zeros(shape=(data.shape[0], np.amax(data)+1))
  for i in range(0, data.shape[0]):
    new_data[i, data[i]] = 1
  return new_data

def create_batches(data, batch_size):
  batches = []
  data_size = data.shape[0]
  number_of_batches = data_size//batch_size
  for index in range(number_of_batches):
    begin = index*batch_size
    end = begin+batch_size
    batch = data[begin:end]
    batches.append(batch)
  if batch_size*number_of_batches<data_size :
    batches.append(data[batch_size*number_of_batches])
  return np.array(batches)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = normalization(x_train)
x_test = normalization(x_test)

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

x_train_batches = (create_batches(x_train, 30))
x_test_batches = (create_batches(x_test, 30))


learning_rate = 0.1

network = mlb.Network()

network.use_losses(mlb.error_functions.minimum_squared_error,mlb.error_functions.minimum_squared_error_derivative)

network.initialize_layers(28*28,12,10,1)


network.fit(x_train_batches, y_train, 1, learning_rate)
predicted = network.predict(x_test_batches)

for i in range(y_test):
    pred = np.dot(predicted[i], y_test[i])