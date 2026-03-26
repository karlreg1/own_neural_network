import numpy as np

class MSELoss():
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred)**2)
    
    def backward(self, y_pred, y_true):
        return 2 * (y_pred - y_true)


class CrossEntropyLoss():
    def forward(self, y_pred, y_true):
        return -np.sum(y_true * np.log(y_pred + 1e-8))
    def backward(self, y_pred, y_true):
        return y_pred - y_true  #combined softmax + crossentropy gradient
