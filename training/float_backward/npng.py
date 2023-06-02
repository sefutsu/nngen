import numpy as np
import nngen as ng
from pathlib import Path
from util import relu, deriv_relu, identity
from common import *

class Node:
    def __init__(self, shape, name, output_addr):
        self.name = name
        self.shape = shape
        self.size = np.cumprod(shape)[-1]
        self.output_addr = output_addr
    def fetch_output_from_buffer(self, buf):
        x = buf[self.output_addr:self.output_addr + self.size].view(np.int8)
        return x.reshape(self.shape).astype(np.float32)

class PlaceHolder(Node):
    def __init__(self, ch, name, output_addr=0):
        super().__init__((1, ch), name, output_addr)
        self.out_ng = ng.placeholder(dtype=act_dtype, shape=self.shape, name=name)
    def forward_np(self, feed_dict):
        return feed_dict[self.name]
    def backward_np(self, delta, w):
        pass
    def forward_ng_eval(self, feed_dict):
        return feed_dict[self.name] / act_scale_factor
    def fetch_output_from_buffer(self, buf):
        return super().fetch_output_from_buffer(buf) / act_scale_factor

class Matmul(Node):
    def __init__(self, input_node, in_dim, out_dim, name, output_addr=0, act=True):
        super().__init__((1, out_dim), name, output_addr)

        self.w = np.random.uniform(low=-0.08, high=0.08,
                                   size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim).astype(np.float32)
        if act:
            self.act = relu
            self.deriv_act = deriv_relu
        else:
            self.act = identity
            self.deriv_act = identity
        
        self.x = None
        self.u = None
        self.dw = 0
        self.db = 0

        self.input_node = input_node
        
        self.w_ng = ng.variable(dtype=weight_dtype, shape=(out_dim, in_dim), name=f"{self.name}.w")
        self.b_ng = ng.variable(dtype=bias_dtype, shape=(out_dim), name=f"{self.name}.b")
        self.s_ng = ng.variable(dtype=scale_dtype, shape=(out_dim), name=f"{self.name}.scale")
        self.vshamt_ng = ng.variable(dtype=ng.int8, shape=(1,), name=f"{self.name}.vshamt")
        self.out_ng = ng.matmul(self.input_node.out_ng, self.w_ng,
            bias=self.b_ng, scale=self.s_ng, rshift_out=self.vshamt_ng, transposed_b=True,
            act_func=ng.relu if act else None, dtype=act_dtype, sum_dtype=bias_dtype, name=f"{self.name}.out")
        self.output_addr = output_addr

    def reset_ng(self):
        self.w_ng.quantized = False
        self.b_ng.quantized = False
        self.s_ng.quantized = False
        self.out_ng.quantized = False

    def forward_np(self, feed_dict):
        self.x = self.input_node.forward_np(feed_dict)
        self.u = np.matmul(self.x, self.w) + self.b
        return self.act(self.u)

    def backward_np(self, delta, w=None):
        if w is None: #出力層
            self.delta = delta
        else:
            self.delta = self.deriv_act(self.u) * np.matmul(delta, w.T)
        self.compute_grad()
        self.input_node.backward_np(self.delta, self.w)
    
    def compute_grad(self):
        batch_size = self.delta.shape[0]
        self.dw += np.matmul(self.x.T, self.delta) / batch_size
        self.db += np.matmul(np.ones(batch_size), self.delta) / batch_size

    def update_params(self, alpha):
        self.w -= alpha * self.dw
        self.b -= alpha * self.db
        self.dw = 0
        self.db = 0
        
    def sync_params(self):
        self.w_ng.set_value(self.w.T)
        self.b_ng.set_value(self.b)
        self.s_ng.set_value(np.ones(self.s_ng.shape))
        self.vshamt_ng.set_value([0])
        
    def forward_ng_eval(self, feed_dict):
        self.x = self.input_node.forward_ng_eval(feed_dict)
        # 本当は間違いだが、backwardのderiv_reluに渡すためだけなら大丈夫
        self.u = ng.eval([self.out_ng], **feed_dict)[0].astype(np.float32) / self.out_ng.scale_factor
        return self.u
    
    def backward_np(self, delta, w=None):
        if w is None: #出力層
            self.delta = delta
        else:
            self.delta = self.deriv_act(self.u) * np.matmul(delta, w.T)
        self.compute_grad()
        self.input_node.backward_np(self.delta, self.w)
    
    def fetch_output_from_buffer(self, buf):
        self.x = self.input_node.fetch_output_from_buffer(buf)
        self.u = super().fetch_output_from_buffer(buf) / self.out_ng.scale_factor
        return self.u
    
    def save_params_np(self, path):
        np.save(Path(path) / (self.name + "_w.npy"), self.w)
        np.save(Path(path) / (self.name + "_b.npy"), self.b)
    def load_params_np(self, path):
        self.w = np.load(Path(path) / (self.name + "_w.npy"))
        self.b = np.load(Path(path) / (self.name + "_b.npy"))
