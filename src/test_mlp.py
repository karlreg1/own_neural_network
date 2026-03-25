import matplotlib.pyplot as plt
import numpy as np
from mlp import MLP, Layer
from activations import Relu, Linear
from losses import MSELoss

#data for training
X = np.random.randn(100, 3)
y_true = X @ np.array([1.5, -2.0, 0.5]) + 0.1

mlp = MLP()
layer1 = Layer(3, 4, Relu())
layer2 = Layer(4, 4, Relu())
layer3 = Layer(4, 1, Linear())
mlp.add_layer(layer1)
mlp.add_layer(layer2)
mlp.add_layer(layer3)

loss_fn = MSELoss()
losses = []
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    epoch_loss = 0
    for x, y in zip(X, y_true):
        y_pred = mlp.forward(x)
        epoch_loss += loss_fn.forward(y_pred, y)
        delta = loss_fn.backward(y_pred, y)
        mlp.backward(delta)
        mlp.update(learning_rate)
    losses.append(epoch_loss / len(X))

plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss over the epochs")
plt.show()