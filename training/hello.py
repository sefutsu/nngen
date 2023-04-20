"""
   NNgen: A Fully-Customizable Hardware Synthesis Compiler for Deep Neural Network

   Copyright 2017, Shinya Takamaeda-Yamazaki and Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import os

import nngen.training as ng


# --------------------
# (1) Represent a DNN model as a dataflow by NNgen operators
# --------------------

# data types
act_dtype = ng.int8
weight_dtype = ng.int8
bias_dtype = ng.int32
scale_dtype = ng.int8
batchsize = 1

# input
input_layer = ng.placeholder(dtype=act_dtype,
                             shape=(batchsize, 32),
                             name='input_layer')

# layer 0: conv2d (with bias and scale (= batchnorm)), relu, max_pool
# w0 = ng.variable(dtype=weight_dtype,
#                  shape=(64, 3, 3, 3),  # Och, Ky, Kx, Ich
#                  name='w0')
# b0 = ng.variable(dtype=bias_dtype,
#                  shape=(w0.shape[0],), name='b0')
# s0 = ng.variable(dtype=scale_dtype,
#                  shape=(w0.shape[0],), name='s0')

# a0 = ng.conv2d(input_layer, w0,
#                strides=(1, 1, 1, 1),
#                bias=b0,
#                scale=s0,
#                act_func=ng.relu,
#                dtype=act_dtype,
#                sum_dtype=ng.int32)

# a0p = ng.max_pool_serial(a0,
#                          ksize=(1, 2, 2, 1),
#                          strides=(1, 2, 2, 1))

# # layer 1: conv2d, relu, reshape
# w1 = ng.variable(weight_dtype,
#                  shape=(64, 3, 3, a0.shape[-1]),
#                  name='w1')
# b1 = ng.variable(bias_dtype,
#                  shape=(w1.shape[0],),
#                  name='b1')
# s1 = ng.variable(scale_dtype,
#                  shape=(w1.shape[0],),
#                  name='s1')

# a1 = ng.conv2d(a0p, w1,
#                strides=(1, 1, 1, 1),
#                bias=b1,
#                scale=s1,
#                act_func=ng.relu,
#                dtype=act_dtype,
#                sum_dtype=ng.int32)

# a1r = ng.reshape(a1, [batchsize, -1])

# layer 2: full-connection, relu
w2 = ng.variable(weight_dtype,
                 shape=(64, input_layer.shape[-1]),
                 name='w2')
b2 = ng.variable(bias_dtype,
                 shape=(w2.shape[0],),
                 name='b2')
s2 = ng.variable(scale_dtype,
                 shape=(w2.shape[0],),
                 name='s2')


a2 = ng.matmul(input_layer, w2,
               bias=b2,
               scale=s2,
               transposed_b=True,
               act_func=ng.relu,
               dtype=act_dtype)

# layer 3: full-connection, relu
w3 = ng.variable(weight_dtype,
                 shape=(10, a2.shape[-1]),
                 name='w3')
b3 = ng.variable(bias_dtype,
                 shape=(w3.shape[0],),
                 name='b3')
s3 = ng.variable(scale_dtype,
                 shape=(w3.shape[0],),
                 name='s3')

# output
output_layer = ng.matmul(a2, w3,
                         bias=b3,
                         scale=s3,
                         transposed_b=True,
                         name='output_layer',
                         dtype=act_dtype)


# --------------------
# (2) Assign weights to the NNgen operators
# --------------------

# In this example, random floating-point values are assigned.
# In a real case, you should assign actual weight values
# obtianed by a training on DNN framework.

# If you don't you NNgen's quantizer, you can assign integer weights to each tensor.


import numpy as np

# w0_value = np.random.normal(size=w0.length).reshape(w0.shape)
# w0_value = np.clip(w0_value, -3.0, 3.0)
# w0.set_value(w0_value)

# b0_value = np.random.normal(size=b0.length).reshape(b0.shape)
# b0_value = np.clip(b0_value, -3.0, 3.0)
# b0.set_value(b0_value)

# s0_value = np.ones(s0.shape)
# s0.set_value(s0_value)

# w1_value = np.random.normal(size=w1.length).reshape(w1.shape)
# w1_value = np.clip(w1_value, -3.0, 3.0)
# w1.set_value(w1_value)

# b1_value = np.random.normal(size=b1.length).reshape(b1.shape)
# b1_value = np.clip(b1_value, -3.0, 3.0)
# b1.set_value(b1_value)

# s1_value = np.ones(s1.shape)
# s1.set_value(s1_value)

w2_value = np.random.normal(size=w2.length).reshape(w2.shape)
w2_value = np.clip(w2_value, -3.0, 3.0)
w2.set_value(w2_value)

b2_value = np.random.normal(size=b2.length).reshape(b2.shape)
b2_value = np.clip(b2_value, -3.0, 3.0)
b2.set_value(b2_value)

s2_value = np.ones(s2.shape)
s2.set_value(s2_value)

w3_value = np.random.normal(size=w3.length).reshape(w3.shape)
w3_value = np.clip(w3_value, -3.0, 3.0)
w3.set_value(w3_value)

b3_value = np.random.normal(size=b3.length).reshape(b3.shape)
b3_value = np.clip(b3_value, -3.0, 3.0)
b3.set_value(b3_value)

s3_value = np.ones(s3.shape)
s3.set_value(s3_value)

# Quantizing the floating-point weights by the NNgen quantizer.
# Alternatively, you can assign integer weights by yourself to each tensor.

imagenet_mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
imagenet_std = np.array([0.229, 0.224, 0.225]).astype(np.float32)

if act_dtype.width > 8:
    act_scale_factor = 128
else:
    act_scale_factor = int(round(2 ** (act_dtype.width - 1) * 0.5))

input_scale_factors = {'input_layer': act_scale_factor}
input_means = {'input_layer': imagenet_mean * act_scale_factor}
input_stds = {'input_layer': imagenet_std * act_scale_factor}

# --------------------
# (4) Verify the DNN model behavior by executing the NNgen dataflow as a software
# --------------------

# In this example, random integer values are assigned.
# In real case, you should assign actual integer activation values, such as an image.

input_layer_value = np.random.normal(size=input_layer.length).reshape(input_layer.shape)
input_layer_value = input_layer_value * imagenet_std + imagenet_mean
input_layer_value = np.clip(input_layer_value, -3.0, 3.0)
input_layer_value = input_layer_value * act_scale_factor
input_layer_value = np.clip(input_layer_value,
                            -1 * 2 ** (act_dtype.width - 1) - 1, 2 ** (act_dtype.width - 1))
input_layer_value = np.round(input_layer_value).astype(np.int64)

eval_outs = ng.eval([output_layer], input_layer=input_layer_value)
output_layer_value = eval_outs[0]

print(output_layer_value)
