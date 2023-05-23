from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from nngen.training.util import Context
import nngen as ng
import nngen.verify as verify
import nngen.verify.backward as backward

input_name = 'input_layer'
act_scale_factor = 32
input_mean = 0
input_std = 1
input_scale_factors = {input_name: act_scale_factor}
input_means = {input_name: input_mean * act_scale_factor}
input_stds = {input_name: input_std * act_scale_factor}

value_low = -3
value_high = 3
in_dim = 3
mid_dim = 10
out_dim = 4

def _relative_error_assert(x, y):
    eps = 0.8
    print(x, y, np.abs(x - y) / np.abs(y).max(), flush=True)
    assert (np.abs(x - y) < eps * np.abs(y).max()).all()

def test_backward():
    x = np.random.normal(input_mean, input_std, (in_dim, mid_dim))
    w = np.random.uniform(value_low, value_high, (out_dim, mid_dim))
    b = np.random.uniform(value_low, value_high, (out_dim,))
    t = np.zeros((in_dim, out_dim), dtype=np.float64)

    xn = ng.placeholder(dtype=ng.int8, shape=x.shape, name=input_name)
    xni = np.round(x * act_scale_factor).astype(np.int64)
    xn.requires_grad = True
    wn = ng.variable(dtype=ng.int8, shape=w.shape)
    wn.set_value(w)
    bn = ng.variable(dtype=ng.int32, shape=b.shape)
    bn.set_value(b)
    tn = ng.variable(dtype=np.float64, shape=t.shape)
    tn.set_value(t)

    an = ng.matmul(xn, wn, bn, transposed_a=False, transposed_b=True, dtype=ng.int8)
    an.requires_grad = True
    rn = ng.mse_loss(an, tn, reduction='sum')
    ng.quantize([an], input_scale_factors, input_means, input_stds)
    ng.eval([rn], **{input_name: xni})
    ng.backward([rn])

    xt = torch.tensor(x, requires_grad=True)
    wt = torch.tensor(w, requires_grad=True)
    bt = torch.tensor(b, requires_grad=True)
    tt = torch.tensor(t)

    at = torch.matmul(xt, wt.T) + bt
    at.retain_grad()
    rt = torch.nn.MSELoss(reduction='sum')(at, tt)
    rt.backward()

    _relative_error_assert(an.grad.astype(np.float64) / an.grad_scale_factor, at.grad.numpy())
    _relative_error_assert(xn.grad.astype(np.float64) / xn.grad_scale_factor, xt.grad.numpy())
    _relative_error_assert(wn.grad.astype(np.float64) / wn.grad_scale_factor, wt.grad.numpy())
    _relative_error_assert(bn.grad.astype(np.float64) / bn.grad_scale_factor, bt.grad.numpy())
    
def test_backward_with_relu():
    x = np.random.normal(input_mean, input_std, (in_dim, mid_dim))
    w = np.random.uniform(value_low, value_high, (out_dim, mid_dim))
    b = np.random.uniform(value_low, value_high, (out_dim,))
    t = np.zeros((in_dim, out_dim), dtype=np.float64)

    xn = ng.placeholder(dtype=ng.int8, shape=x.shape, name=input_name)
    xn.requires_grad = True
    xni = np.round(x * act_scale_factor).astype(np.int64)
    wn = ng.variable(dtype=ng.int8, shape=w.shape)
    wn.set_value(w)
    bn = ng.variable(dtype=ng.int32, shape=b.shape)
    bn.set_value(b)
    tn = ng.variable(dtype=np.float64, shape=t.shape)
    tn.set_value(t)

    an = ng.matmul(xn, wn, bn, transposed_a=False, transposed_b=True, dtype=ng.int8, act_func=ng.relu)
    an.requires_grad = True
    rn = ng.mse_loss(an, tn, reduction='sum')
    ng.quantize([an], input_scale_factors, input_means, input_stds)
    ng.eval([rn], **{input_name: xni})
    ng.backward([rn])

    xt = torch.tensor(x, requires_grad=True)
    wt = torch.tensor(w, requires_grad=True)
    bt = torch.tensor(b, requires_grad=True)
    tt = torch.tensor(t)

    at = nn.ReLU()(torch.matmul(xt, wt.T) + bt)
    at.retain_grad()
    rt = torch.nn.MSELoss(reduction='sum')(at, tt)
    rt.backward()

    _relative_error_assert(an.grad.astype(np.float64) / an.grad_scale_factor, at.grad.numpy())
    _relative_error_assert(xn.grad.astype(np.float64) / xn.grad_scale_factor, xt.grad.numpy())
    _relative_error_assert(wn.grad.astype(np.float64) / wn.grad_scale_factor, wt.grad.numpy())
    _relative_error_assert(bn.grad.astype(np.float64) / bn.grad_scale_factor, bt.grad.numpy())
    
