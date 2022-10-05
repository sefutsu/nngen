from __future__ import absolute_import
from __future__ import print_function

import nngen as ng
import numpy as np
import torch

def generate_inputs():
    return [
        np.random.randint(-100, 100, size=(2, 3), dtype=np.int64)
    ]

def generate_weights():
    return {
        "w0": np.random.randint(-100, 100, size=(2, 3), dtype=np.int64),
        "w1": np.random.randint(-100, 100, size=(3, 2), dtype=np.int64),
        "b0": np.random.randint(-100, 100, size=(2,), dtype=np.int64),
        "b1": np.random.randint(-100, 100, size=(3,), dtype=np.int64),
        "scale0": np.random.randint(-100, 100, size=(2,), dtype=np.int64),
        "scale1": np.random.randint(-100, 100, size=(3,), dtype=np.int64),
    }

def nngen_gradient(inputs, weights):
    dtype = ng.int32

    input_layer = ng.placeholder(dtype=dtype, shape=(2, 3), name="input_layer")
    w0 = ng.variable(dtype=dtype, shape=(2, 3), name="w0")
    b0 = ng.variable(dtype=dtype, shape=(2,), name="b0")
    scale0 = ng.variable(dtype=dtype, shape=(2,), name="scale0")
    s0 = ng.matmul(input_layer, w0, bias=b0, transposed_b=True, scale=scale0, act_func=ng.relu, dtype=dtype)

    w1 = ng.variable(dtype=dtype, shape=(3, 2), name="w1")
    b1 = ng.variable(dtype=dtype, shape=(3,), name="b1")
    scale1 = ng.variable(dtype=dtype, shape=(3,), name="scale1")
    s1 = ng.matmul(s0, w1, bias=b1, transposed_b=True, scale=scale1, act_func=ng.relu, dtype=dtype)

    w0.set_value(weights["w0"])
    w1.set_value(weights["w1"])
    b0.set_value(weights["b0"])
    b1.set_value(weights["b1"])
    scale0.set_value(weights["scale0"])
    scale1.set_value(weights["scale1"])

    eval_res = ng.eval([s1], input_layer=inputs[0])
    return ng.gradient(s1, input_layer)


def torch_gradient(inputs, weights):
    dtype = torch.double

    input_value = torch.tensor(inputs[0], dtype=dtype, requires_grad=True)
    w0 = torch.tensor(weights["w0"], dtype=dtype)
    b0 = torch.tensor(weights["b0"], dtype=dtype)
    w1 = torch.tensor(weights["w1"], dtype=dtype)
    b1 = torch.tensor(weights["b1"], dtype=dtype)
    scale0 = torch.tensor(weights["scale0"], dtype=dtype)
    scale1 = torch.tensor(weights["scale1"], dtype=dtype)

    s0 = torch.relu((input_value @ w0.T + b0) * scale0)
    s1 = torch.relu((s0 @ w1.T + b1) * scale1)
    s = s1.sum()

    s.backward()

    return input_value.grad.numpy()
