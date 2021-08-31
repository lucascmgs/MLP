import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import sklearn.metrics as mt
import sys
import time 

from MLPLucasCordeiroMarques import *
autor = "Lucas Cordeiro Marques"

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0],28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

#y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0],28*28)
x_test = x_test.astype('float32')
x_test /= 255
#y_test = np_utils.to_categorical(y_test)

def one_hot(vet):
    n = vet.shape[0]
    mat = np.zeros(shape=(n,10),dtype=np.int32)
    for i in range(n):
        j = vet[i]
        mat[i,j]=1
    return mat

y_train_one_hot = one_hot(y_train)
y_test_one_hot = one_hot(y_test)

np.random.seed(13)  # garante que todas as inicializações são iguais
start = time.perf_counter()

net = Network()
net.layer(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.layer(ActLayer(tanh, tanh_prime))
net.layer(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.layer(ActLayer(tanh, tanh_prime))
net.layer(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.layer(ActLayer(tanh, tanh_prime))
net.use(mse, mse_prime)

err_train=net.fit(x_train, y_train_one_hot, epochs=100, mini_batch=1000, learning_rate=0.1)

stop=time.perf_counter()
time_train = stop-start
print(f'tempo de treinamento = {time_train:.2f} s')

