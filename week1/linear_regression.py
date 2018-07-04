import numpy as np
import matplotlib.pyplot as plt

class LinearRegressor:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self._errors = []
        self.theta = np.random.rand(x.shape[1])
        self.m = x.shape[1]
  
    def h(self, x):
        return np.dot(self.theta, x)

    def gradient_descent(self, learning_rate=0.01):
        _theta = []
        for j, t in enumerate(self.theta):
            gradient = np.sum((self.h(x) - y)*x[j] for x, y in zip(self.x, self.y))
            new = t - learning_rate / self.m * gradient
            _theta.append(new)
        return np.array(_theta)
      
    def error(self):
        return np.sum([(self.h(x) - y)**2 for x, y in zip(self.x, self.y)]) / (2 *self.m)

    def predict(self, x):
        return self.h(x)

    def train(self, epochs=100, learning_rate=0.01):
        for n in range(epochs):
            self.theta = self.gradient_descent(learning_rate)
            err = self.error()
            self._errors.append(err)

    def plot(self):
        plt.plot(self._errors)
        plt.show()


def h(theta, x):
    return np.dot(theta, x)

def _gradient_descent(data, theta, alpha=0.01):
    return np.array([t - (alpha / len(data)) * np.sum([(h(theta, x) - y)*x[i] for x, y in data]) for i, t in enumerate(theta)])


def gradient_descent(x, y, theta, alpha=0.01):
    _theta = []
    for i, t in enumerate(theta):
        gradient = np.sum((h(theta, x) - y)*x[i] for x, y in zip(x, y))
        new = t - alpha / len(x) * gradient
        _theta.append(new)
    return np.array(_theta)


def gradient_descent_vectorization(x, y, theta, alpha=0.01):    
    return theta - alpha / len(x) * np.dot(x.T, h(x, theta) - y)


def error(theta, x, y):
    return np.sum([(h(theta, x) - y)**2 for x, y in zip(x, y)]) / (2 *len(x))


data = np.array([[1, 1], [1, 2], [1, 4], [1, 8]])
y = [1, 2, 4, 8]
theta = np.random.rand(2)
errors = []
#data = np.hstack((np.ones((len(data), 1)), data))
print(theta)
for n in range(1000):
    theta = gradient_descent_vectorization(data, y, theta, 0.008)
    err = error(theta, data, y)
    # if err < 0.001:
    #     print('Finish at', n)
    #     break
    errors.append(err)
    print(theta)
print(err)
plt.plot(errors)
plt.show()

print('test:', )
