import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self):
        self.w = None

    def predict(self, x):
        return 1 if np.dot(self.w, x) > 0 else -1

    def fit(self, X, Y, l_rate, epochs):
        self.w = np.random.randn(X.shape[1] + 1) * 0.01  
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X]) 
        for _ in range(epochs):
            for i in range(len(X)):
                y_hat = self.predict(X_bias[i])
                self.w = self.w + l_rate * (Y[i] - y_hat) * X_bias[i]

    def plot_perceptron(self, X, Y):
        for i in range(len(X)):
            if Y[i] == 1:
                plt.scatter(X[i][0], X[i][1], color='blue', marker='o', s=100)
            else:
                plt.scatter(X[i][0], X[i][1], color='red', marker='x', s=100)
        x1 = np.linspace(-0.5, 1.5, 100)
        x2 = (-self.w[0] - self.w[1] * x1) / self.w[2]

        plt.plot(x1, x2, color='green')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True)
        plt.show()



X = np.array([
    [0.9, 0.8],
    [1.0, 1.0],
    [0.8, 0.9],
    [0.7, 0.8],
    [0.9, 0.7],
    [0.0, 0.0],
    [0.1, 0.2],
    [0.2, 0.1],
    [0.0, 0.9],
    [0.9, 0.0],
    [0.1, 0.8],
    [0.8, 0.1],
    [0.2, 0.3],
    [0.3, 0.2],
    [0.0, 0.5],
    [0.5, 0.0],
    [0.1, 0.4],
    [0.4, 0.1],
    [0.2, 0.0],
    [0.0, 0.2],
])

Y = np.array([1, 1, 1, 1, 1,
             -1,-1,-1,-1,-1,
             -1,-1,-1,-1,-1,
             -1,-1,-1,-1,-1])

p = Perceptron()
p.fit(X, Y, l_rate=0.1, epochs=2)

# should be +1
print(p.predict([1, 1, 1]))
# should be -1
print(p.predict([1, 0, 0]))

p.plot_perceptron(X, Y)