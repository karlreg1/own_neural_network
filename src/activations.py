import numpy as np

class Relu():
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x):
        #converts the matrix x first into boolean values (x_i > 1 -> true else false) and converts them then into folating point numbers
        return (x > 0).astype(float)
    
class LeakyRelu():
    def __init__(self, beta=0.01):
        self.beta = beta
    def forward(self, x):
        return np.where(x > 0, x, self.beta * x)

    def backward(self, x):
        return np.where(x > 0, 1, self.beta)

class Linear():
    def forward(self, x):
        return x
    
    def backward(self, x):
        return np.ones_like(x)
    
class SoftMax():
    def forward(self, x):
        exp = np.exp(x - np.max(x))
        self.output = exp / exp.sum()
        return self.output
    
    def backward(self, x):
        #shortcut for better backward pass
        return np.ones_like(x)
