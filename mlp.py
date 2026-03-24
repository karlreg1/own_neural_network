import numpy as np
import matplotlib as plt

class Layer():
    def __init__(self, input_size, output_size, activation_fct):
        self.activation_fct = activation_fct
        self.weights = np.random.randn(output_size, input_size) #weights are initialized randomly
        self.biases = np.zeros(output_size)

    #calculates the whole activation for layer
    def forward(self, x):
        self.x = x
        self.pre_activation = np.dot(self.weights, x) + self.biases
        activation = self.activation_fct.forward(self.pre_activation)
        return activation
    
    def backward(self, delta):
        delta_activation = self.activation_fct.backward(self.pre_activation) * delta
        self.grad_weights = np.outer(delta_activation, self.x)
        self.grad_biases = delta_activation
        grad_input = self.weights.T @ delta_activation
    
        return grad_input



class Relu():
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        #converts the matrix x first into boolean values (x_i > 1 -> true else false) and converts them then into folating point numbers
        return (x > 0).astype(float)

class MSELoss():
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred)**2)
    
    def backward(self, y_pred, y_true):
        return (2 / len(y_true)) * (y_pred - y_true)

class MLP():
    def __init__(self):
        self.layers = []

    def add_layer(self, layer:Layer):
        self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, delta):
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

mlp = MLP()
layer1 = Layer(3, 4, Relu())
layer2 = Layer(4, 4, Relu())
layer3 = Layer(4, 1, Relu())
mlp.add_layer(layer1)
mlp.add_layer(layer2)
mlp.add_layer(layer3)

x = np.array([1.0, 2.0, 3.0])  
result = mlp.forward(x)
print(result)