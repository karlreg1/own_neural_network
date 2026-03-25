import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from mlp import MLP, Layer
from activations import Relu, SoftMax
from losses import CrossEntropyLoss
from utils import OneHotEncoder

mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.to_numpy() / 255.0
Y = mnist.target.astype(int).to_numpy()

encoder = OneHotEncoder()

Y = encoder.encode(Y, num_classes=10)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = Y[:60000], Y[60000:]

mlp = MLP()
mlp.add_layer(Layer(784, 128, Relu()))
mlp.add_layer(Layer(128, 64, Relu()))
mlp.add_layer(Layer(64, 10, SoftMax()))

loss_fn = CrossEntropyLoss()
learning_rate = 0.01
