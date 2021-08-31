import numpy as np
import time


def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def sigmoid_prime(x):
    term = 1/(1+np.exp(-x))
    return term*(1-term)

def relu_prime(x):
    return (x>0).astype(x.dtype)

def mse( target_y, obtained_y):
    return np.mean(np.power(target_y-obtained_y, 2))

#Retorna um vetor, cada elemento dividido pelo numero de elementos do vetor
def mse_prime(target_y, obtained_y):
    size = target_y.shape[0]
    return 2*(obtained_y - target_y)/size

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




class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self, input):
        return

    def backward(self, gradient_output, learning_rate):
        return

class ActLayer(Layer):
    
    def __init__(self, given_activation_function, given_activation_function_prime):
        self.activation_function = given_activation_function
        self.activation_function_prime = given_activation_function_prime

    def forward(self, given_input):
        #Guardamos o input
        self.input = given_input
        #Aplicamos a função de ativação no input
        self.output = self.activation_function(given_input)
        #Passamos o output adiante
        
        return self.output

    #Retornamos dE/dx a partir do gradiente do output, segundo a fórmula:
    #dE/dxj = derivada_ativacao(xj)*dE/dyj, onde dE/dyj é o gradient_output
    def backward(self, gradient_output, learning_rate):
        return self.activation_function_prime(self.input)*gradient_output




class FCLayer(Layer):
    def __init__(self, input_size, output_size) -> None:
        #Retornam sempre valores entre -1 e 1. Os sizes aqui são as dimensões do vetor retornado (a função retorna um escalar caso não os tenha na chamada)
        self.weights = 2*np.random.rand(input_size, output_size) - 1.0
        self.bias = 2*np.random.rand(1, output_size) - 1.0

    def forward(self, given_input):
        # Guardamos o input
        self.input = given_input
        # Fazemos uma combinação linear, aplicando os pesos a cada entrada correspondente, somando os termos e guardando o resultado
        self.output = np.dot(given_input, self.weights) + self.bias

        # Passamos o resultado adiante
        return self.output

    # Importante notar que dE/dX (gradient_input) é um vetor, enquanto dE/dW (gradient_weights) é uma matriz
    def backward(self, gradient_output, learning_rate):
        gradient_input = np.dot(gradient_output, self.weights.T)
        gradient_weights = np.dot(self.input.T,gradient_output)
        gradient_bias = np.sum(gradient_output,axis=0).reshape((1,-1))

        self.weights -= gradient_weights*learning_rate
        self.bias -= gradient_bias*learning_rate
        return gradient_input

class Network:
    def __init__(self) -> None:
        self.layers = []
        self.loss_function = None
        self.loss_function_prime = None

    def layer(self, layer):
        self.layers.append(layer)

    def use(self, given_loss_function, given_loss_function_prime):
        self.loss_function = given_loss_function
        self.loss_function_prime = given_loss_function_prime
    
    def predict(self, input_data):
        number_of_samples = len(input_data)
        result = []
        for i in range(number_of_samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            result.append(output)

        
        return np.array(result)

    def fit(self, x_train, y_train, epochs, mini_batch, learning_rate):
        if mini_batch > 0:
            x_train = create_batches(x_train, mini_batch)

        number_of_samples = x_train.shape[0]
        

        errors_by_epoch = np.array([])

        for i in range(epochs):
            error_amount = 0
            for j in range(number_of_samples):
                output = x_train[j]
                for lay in self.layers:
                    output = lay.forward(output)
                
                error_amount += self.loss_function( y_train[j], output)
                gradient_output = self.loss_function_prime(y_train[j], output)
                
                for lay in reversed(self.layers):
                    gradient_output = lay.backward(gradient_output, learning_rate)

            error_amount = error_amount/number_of_samples
            print(f"Época:{i+1}, Erro: {error_amount}")
            errors_by_epoch = np.append(errors_by_epoch, error_amount)
        return errors_by_epoch

       

