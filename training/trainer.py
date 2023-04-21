from pynq import Overlay, allocate
import numpy as np

import nngen_ctrl as ngc

class Trainer:
    def __init__(self, model, name, ip_name=None, global_buffer_size=100*1024):
        bitfile_name = name + ".bit"
        if ip_name is None:
            ip_name = name + "_0"
        overlay = Overlay(bitfile_name)
        self.ip = ngc.nngen_core(overlay, ip_name)

        self.model = model
        self.global_buffer = allocate(shape=(global_buffer_size,), dtype=np.uint8)
        self.ip.set_global_buffer(self.global_buffer)

    def read_global_buffer(self, addr_begin, length):
        return self.global_buffer[addr_begin:addr_begin+length].view(np.int8)

    def write_global_buffer(self, addr_begin, length, value):
        # TODO: boundary check
        self.global_buffer[addr_begin:addr_begin+length] = value.view(np.uint8)

    def load_address_info(self, file):
        # TODO
        self.output_offset = 0
        self.input_offset = 64
        self.param_offset = 896
        self.input_size = 784
        self.output_size = 10
        self.param_size = 90944
        
        info = {} # TODO
        self.model.load_address_info(info, self.read_global_buffer, self.write_global_buffer)
        # TODO: Make partial functions `read` and `write` for each node

    def load_params(self, file):
        param = np.load(file)
        if file.endswith(".npz"):
            param = param[0]
        self.write_global_buffer(self.param_offset, self.param_size, param)
