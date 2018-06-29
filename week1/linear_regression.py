import numpy as np
import matplotlib.pyplot as plt

# class Hypothesis:
    
#     def foward(self):
#         raise NotImplementedError
    
#     def backward(self):
#         raise NotImplementedError


# class LinOneDim(Hypothesis):
#     def __init__(self, t0, t1, alpha):
#         self.t0 = t0
#         self.t1 = t1
#         self.alpha = alpha

#     def foward(self, x):
#         return self.t0 + self.t1 * x

#     def backward(self):
#         self.t0 = self-t0 - self.alpha
  
def h(theta, x):
    return np.dot(theta, x)

# def gradient_descent(data, t0=0, t1=1, alpha=0.01):
#     t0 = t0 - alpha * sum([h(x, t0, t1) - y for x, y in data]) / len(data)
#     t1 = t1 - alpha * sum([(h(x, t0, t1) - y) * x for x, y in data]) / len(data)
#     return t0, t1

def _gradient_descent(data, theta, alpha=0.01):
    return np.array([t - (alpha / len(data)) * np.sum([(h(theta, x) - y)*x[i] for x, y in data]) for i, t in enumerate(theta)])


def gradient_descent(x, y, theta, alpha=0.01):
    _theta = []
    for i, t in enumerate(theta):
        gradient = (alpha / len(data)) * np.sum((h(theta, x) - y)*x[i] for x, y in zip(x, y))
        new = t - (alpha / len(data)) * gradient
        _theta.append(new)
    return np.array(_theta)

def error(theta, x, y):
    return np.sum([(h(theta, x) - y)**2 for x, y in zip(x, y)]) / (2 *len(x))


data = np.array([[1, 1], [1, 2], [1, 4], [1, 8]])
y = [1, 2, 4, 8]
theta = np.random.rand(2)
errors = []
#data = np.hstack((np.ones((len(data), 1)), data))
print(theta)
for n in range(1000):
    theta = gradient_descent(data, y, theta, 0.3)
    err = error(theta, data, y)
    # if err < 0.001:
    #     print('Finish at', n)
    #     break
    errors.append(err)
    print(theta)
print(err)
plt.plot(errors)
plt.show()