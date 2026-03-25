import numpy as np

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
    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases

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

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

