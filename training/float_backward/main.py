from time import time
import numpy as np
from sklearn.utils import shuffle

import nngen as ng
print(ng.__version__)

from trainer import Trainer

trainer = Trainer()

# Training Data
mnist = np.load("mnist.npz")

### How to generate data
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_openml

# mnist = fetch_openml('mnist_784', parser="auto")

# x_mnist = mnist.data.to_numpy().astype(float) / 255.
# t_mnist = mnist.target.to_numpy().astype(int)

# x0_mnist = x_mnist[t_mnist == 0]
# x9_mnist = x_mnist[t_mnist != 0]
# t0_mnist = t_mnist[t_mnist == 0]
# t9_mnist = t_mnist[t_mnist != 0]

# t0_mnist = np.eye(10)[t0_mnist]
# t9_mnist = np.eye(10)[t9_mnist]

# x0_train_mnist, x0_valid_mnist, t0_train_mnist, t0_valid_mnist = train_test_split(x0_mnist, t0_mnist, test_size=0.3)
# x9_train_mnist, x9_valid_mnist, t9_train_mnist, t9_valid_mnist = train_test_split(x9_mnist, t9_mnist, test_size=0.3)

# # Use 1000 data for train and validation
# x0_valid_mnist = x0_valid_mnist[:100]
# x9_valid_mnist = x9_valid_mnist[:900]
# t0_valid_mnist = t0_valid_mnist[:100]
# t9_valid_mnist = t9_valid_mnist[:900]

# x_train_mnist = np.concatenate([x0_train_mnist[:100], x9_train_mnist[:900]])
# t_train_mnist = np.concatenate([t0_train_mnist[:100], t9_train_mnist[:900]])

x0_valid_mnist = mnist["x0_valid"]
x9_valid_mnist = mnist["x9_valid"]
t0_valid_mnist = mnist["t0_valid"]
t9_valid_mnist = mnist["t9_valid"]

x_train_mnist = mnist["x_train"]
t_train_mnist = mnist["t_train"]

def valid(valid_func, inputs, labels):
    correct = 0
    total_cost = 0
    for (x, t) in zip(inputs, labels):
        cost, y_pred = valid_func(x[None], t[None])
        total_cost += cost
        correct += y_pred.argmax() == t.argmax()
    correct /= len(inputs)
    total_cost /= len(inputs)
    return total_cost, correct

def report_accuracy(valid_func, epoch=None, cost=None):
    global x0_valid_mnist, t0_valid_mnist, x9_valid_mnist, t9_valid_mnist
    if epoch is not None:
        print("Epoch:", epoch, end=", ")
    if cost is not None:
        print("Cost: {:.3f}".format(cost), end=", ")
    _, accuracy0 = valid(valid_func, x0_valid_mnist, t0_valid_mnist)
    _, accuracy9 = valid(valid_func, x9_valid_mnist, t9_valid_mnist)
    accuracy = (accuracy0 * len(x0_valid_mnist) + accuracy9 * len(x9_valid_mnist)) / (len(x0_valid_mnist) + len(x9_valid_mnist))
    print("Accuracy0: {:.3f}, Accuracy9: {:.3f}, Accuracy: {:.3f}".format(accuracy0, accuracy9, accuracy))

lr = 0.01

def int_float():
    global x_train_mnist, t_train_mnist
    start_time = time()
    for epoch in range(5):
        x_train_mnist, t_train_mnist = shuffle(x_train_mnist, t_train_mnist)
        train_cost = 0
        for i, (x, t) in enumerate(zip(x_train_mnist, t_train_mnist)):
            cost = trainer.train_ng(x[None, :], t[None, :])
            train_cost += cost
            trainer.update_params(lr)
            trainer.sync_qunatize()
        train_cost /= len(x_train_mnist)
        report_accuracy(trainer.valid_ng, epoch + 1, train_cost)
        
    end_time = time()
    print("Elapsed Time: {:.3f} s".format(end_time - start_time))

def float_float():
    global x_train_mnist, t_train_mnist
    start_time = time()
    for epoch in range(5):
        x_train_mnist, t_train_mnist = shuffle(x_train_mnist, t_train_mnist)
        train_cost = 0
        for i, (x, t) in enumerate(zip(x_train_mnist, t_train_mnist)):
            cost = trainer.train_np(x[None, :], t[None, :])
            train_cost += cost
            trainer.update_params(lr)
        train_cost /= len(x_train_mnist)
        report_accuracy(trainer.valid_np, epoch + 1, train_cost)
        
    end_time = time()
    print("Elapsed Time: {:.3f} s".format(end_time - start_time))

if __name__ == '__main__':
    int_float()
