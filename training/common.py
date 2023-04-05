import nngen as ng

mnist_mean = 0.13090527
mnist_std = 0.30837968

# data types
act_dtype = ng.int8
weight_dtype = ng.int8
bias_dtype = ng.int32
scale_dtype = ng.int8

if act_dtype.width > 8:
    act_scale_factor = 128
else:
    act_scale_factor = int(round(2 ** (act_dtype.width - 1) * 0.5))

input_layer_name = "input_layer"
output_layer_name = "output_layer"

input_scale_factors = {input_layer_name: act_scale_factor}
input_means = {input_layer_name: mnist_mean * act_scale_factor}
input_stds = {input_layer_name: mnist_std * act_scale_factor}
