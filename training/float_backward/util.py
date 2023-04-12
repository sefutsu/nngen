import numpy as np

def sigmoid(x):
    return np.tanh(x * 0.5) * 0.5 + 0.5

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(x, 0)

def deriv_relu(x):
    return (x > 0).astype(x.dtype)

def tanh(x):
    return np.tanh(x)

def deriv_tanh(x):
    return 1 - np.tanh(x)**2

# avoid log(0)
def np_log(x):
    return np.log(np.clip(x, 1e-8, None))


def softmax(x):
    x -= x.max(axis=1, keepdims=True)
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

def deriv_softmax(x):
    return softmax(x) * (1 - softmax(x))

def identity(x):
    return x[:]

def cost_func(x, y, t):
    return (- t * np_log(y)).sum(axis=1).mean()
