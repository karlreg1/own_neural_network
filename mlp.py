import numpy as np
import matplotlib as plt

class Layer():
    def __init__(self, input_size, output_size, activation_fct):
        self.activation_fct = activation_fct
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros(output_size)

    def forward(self, x):
        pre_activation = np.dot(self.weights, x) + self.biases
        activation = self.activation_fct.forward(pre_activation)
        return activation


class Relu():
    def forward(self, x):
        return np.maximum(0, x)


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

    def backward(self):
        pass 

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