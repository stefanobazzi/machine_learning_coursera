from numpy import dot, exp, sum, log


def h(x, theta):
    return dot(x, theta)

def sigmoid(z):
    return 1 / (1 + exp(-z))


def cost_function(theta, x, y):
    m = len(y)
    h = sigmoid(dot(theta, x))
    return 1 / m * np.sum(-y * log(h) - (1 -y) * log(1 - h))

def gradient(theta, x, y):
    m = len(y)
    h = sigmoid(dot(theta, x))
    return 1 / m * sum((h - y) * x)

def logistic_regression():
    pass


def cost_function_reg(theta, x, y, l):
    m = len(y)
    h = sigmoid(dot(theta, x))
    return 1 / m * np.sum(-y * log(h) - (1 -y) * log(1 - h)) + l / (2*m) * sum(theta[2:]**2)

def gradient_reg(theta, x, y, l):
    m = len(y)
    h = sigmoid(dot(theta, x))
    g0 = 1 / m * sum((h - y) * x[:, 0])
    gj = 1 / m * sum((h - y) * x[:, 2:]) + l / m * theta[2:].T
    return np.vstack(g0, gj)