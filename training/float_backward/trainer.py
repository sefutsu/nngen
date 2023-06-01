from npng import PlaceHolder, Matmul, input_scale_factors, input_means, input_stds
from util import softmax, cost_func
from common import *

# https://github.com/opencv/opencv/issues/14884#:~:text=Mar%2017%2C%202020-,I%20had%20a%20similar%20issue%20where,-I%20could%20import
import cv2

import numpy as np
import nngen as ng

from pynq import Overlay, allocate
import nngen_ctrl as ngc

class Trainer:
    def __init__(self):
        # Global buffer上のアドレスを直接埋め込み
        self.l0 = PlaceHolder(784, input_layer_name, 64)
        self.l1 = Matmul(self.l0, 784, 100, "l1", 91840)
        self.l2 = Matmul(self.l1, 100, 100, "l2", 91968)
        self.l3 = Matmul(self.l2, 100, 10, output_layer_name, 0, False)
        self.layers = [self.l1, self.l2, self.l3]

        ng.to_veriloggen([self.l3.out_ng], "mlp", config={"maxi_datawidth": 32}, silent=True)

        bitfile = 'mlp.bit'
        ipname = 'mlp_0'
        memory_size = 100 * 1024

        overlay = Overlay(bitfile)
        self.ip = ngc.nngen_core(overlay, ipname)
        self.buf = allocate(shape=(memory_size,), dtype=np.uint8)
        self.ip.set_global_buffer(self.buf)

        self.output_offset = 0
        self.input_offset = 64
        self.param_offset = 896
        self.input_size = 784
        self.output_size = 10
        self.param_size = 90944

        self.load_params_np("./mnist1-9_weights")
        self.sync_qunatize()

    def load_params_np(self, path):
        for layer in self.layers:
            layer.load_params_np(path)
    def save_params_np(self, path):
        for layer in self.layers:
            layer.save_params_np(path)

    def preprocess_ng(self, x):
        vact = x * act_scale_factor
        vact = np.clip(vact,
                        -1.0 * (2 ** (act_dtype.width - 1) - 1),
                        1.0 * (2 ** (act_dtype.width - 1) - 1))
        vact = np.round(vact).astype(np.int64)
        return vact
    
    def update_params(self, alpha):
        for layer in self.layers:
            layer.update_params(alpha)

    def valid_np(self, x, t):
        y = self.l3.forward_np({input_layer_name: x})
        y = softmax(y)
        cost = cost_func(x, y, t)
        return cost, y
    def train_np(self, x, t):
        cost, y = self.valid_np(x, t)
        delta = y - t
        self.l3.backward_np(delta)
        return cost

    def valid_ng_eval(self, x, t):
        x = self.preprocess_ng(x)
        y = self.l3.forward_ng_eval({input_layer_name: x})
        y = softmax(y)
        cost = cost_func(x, y, t)
        return cost, y
    def train_ng_eval(self, x, t):
        cost, y = self.valid_ng_eval(x, t)
        delta = y - t
        self.l3.backward_np(delta)
        return cost

    def sync_qunatize(self):
        for layer in self.layers:
            layer.reset_ng()
            layer.sync_params()
        ng.quantize([self.l3.out_ng], input_scale_factors, input_means, input_stds)
        # for layer in self.layers:
        #     layer.vshamt_ng.set_value([layer.out_ng.cshamt_out])
        #     layer.out_ng.cshamt_out = 0

        exported_params = ng.export_ndarray(self.l3.out_ng)
        self.buf[self.param_offset:self.param_offset + self.param_size] = exported_params.view(np.uint8)

    def valid_ng(self, x, t):
        x = np.reshape(x, [-1])
        x = self.preprocess_ng(x).astype(np.int8)
        self.buf[self.input_offset:self.input_offset + self.input_size] = x.view(np.uint8)

        self.ip.run()
        self.ip.wait()

        y = self.l3.fetch_output_from_buffer(self.buf)
        y = softmax(y)
        cost = cost_func(x, y, t)
        return cost, y
    def train_ng(self, x, t):
        cost, y = self.valid_ng(x, t)
        delta = y - t
        self.l3.backward_np(delta)
        return cost
