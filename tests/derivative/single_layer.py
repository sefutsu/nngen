from __future__ import absolute_import
from __future__ import print_function
import re

import nngen as ng
import numpy as np
import torch

def generate_inputs():
    return [
        np.random.randint(-100, 100, size=(1, 3), dtype=np.int64)
    ]

def generate_weights():
    return [
        np.random.randint(-100, 100, size=(2, 3), dtype=np.int64),
        np.random.randint(-100, 100, size=(1, 2), dtype=np.int64),
        np.random.randint(-100, 100, size=(2,), dtype=np.int64),
    ]

def nngen_gradient(inputs, weights):
    dtype = ng.int16
    input_layer = ng.placeholder(dtype=dtype, shape=(1, 3), name="input_layer")
    w0 = ng.variable(dtype=dtype, shape=(2, 3), name="w0")
    b0 = ng.variable(dtype=dtype, shape=(2,), name="b0")
    scale = ng.variable(dtype=dtype, shape=(2,), name="scale")
    s0 = ng.matmul(input_layer, w0, bias=b0, scale=scale, transposed_b=True, act_func=ng.relu, dtype=dtype)

    w0.set_value(weights[0])
    b0.set_value(weights[1])
    scale.set_value(weights[2])

    input_value = inputs[0]

    eval_res = ng.eval([s0], input_layer=input_value)
    return ng.gradient(s0, input_layer)


def torch_gradient(inputs, weights):
    dtype = torch.double

    w0 = torch.tensor(weights[0], dtype=dtype)
    b0 = torch.tensor(weights[1], dtype=dtype)
    scale = torch.tensor(weights[2], dtype=dtype)

    input_value = torch.tensor(inputs[0], dtype=dtype, requires_grad=True)
    a = input_value @ w0.T + b0
    k = a * scale
    h = torch.relu(k)
    s = h.sum()

    s.backward()
    return input_value.grad.numpy()
