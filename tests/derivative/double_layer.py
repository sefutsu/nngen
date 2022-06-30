from __future__ import absolute_import
from __future__ import print_function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

import nngen as ng
import numpy as np
import torch

### nngen.verify

dtype = ng.int64

input_layer = ng.placeholder(dtype=dtype, shape=(2, 3), name="input_layer")
w0 = ng.variable(dtype=dtype, shape=(2, 3), name="w0")
b0 = ng.variable(dtype=dtype, shape=(2,), name="b0")
scale0 = ng.variable(dtype=dtype, shape=(2,), name="scale0")
s0 = ng.matmul(input_layer, w0, bias=b0, transposed_b=True, scale=scale0, act_func=ng.relu, dtype=dtype)

w1 = ng.variable(dtype=dtype, shape=(3, 2), name="w1")
b1 = ng.variable(dtype=dtype, shape=(3,), name="b1")
scale1 = ng.variable(dtype=dtype, shape=(3,), name="scale1")
s1 = ng.matmul(s0, w1, bias=b1, transposed_b=True, scale=scale1, act_func=ng.relu, dtype=dtype)

w0_value = np.random.randint(-100, 100, size=(2, 3), dtype=np.int64)
w1_value = np.random.randint(-100, 100, size=(3, 2), dtype=np.int64)
w0.set_value(w0_value)
w1.set_value(w1_value)

b0_value = np.random.randint(-100, 100, size=(2,), dtype=np.int64)
b1_value = np.random.randint(-100, 100, size=(3,), dtype=np.int64)
b0.set_value(b0_value)
b1.set_value(b1_value)

scale0_value = np.random.randint(0, 100, size=(2,), dtype=np.int64)
scale1_value = np.random.randint(0, 100, size=(3,), dtype=np.int64)
scale0.set_value(scale0_value)
scale1.set_value(scale1_value)

input_value = np.random.randint(-100, 100, size=(2, 3), dtype=np.int64)

eval_res = ng.eval([s1], input_layer=input_value)
res = ng.gradient(s1, input_layer)

### torch.autograd

dtype = torch.double

input_value = torch.tensor(input_value, dtype=dtype, requires_grad=True)
w0 = torch.tensor(w0_value, dtype=dtype)
b0 = torch.tensor(b0_value, dtype=dtype)
w1 = torch.tensor(w1_value, dtype=dtype)
b1 = torch.tensor(b1_value, dtype=dtype)

s0 = torch.relu(input_value @ w0.T + b0)
s1 = torch.relu(s0 @ w1.T + b1)
s = s1.sum()

s.backward()

torch_res = input_value.grad.numpy()

if (abs(res - torch_res) < 1e-5).all():
    print("# verify: PASSED")
else:
    print("# verify: FAILED")
    print("Result:", res)
    print("PyTorch Result:", torch_res)

