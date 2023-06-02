import sys
import nngen as ng
    


act_dtype = ng.int8
weight_dtype = ng.int8
sum_dtype = ng.int32
scale_dtype = ng.int8
batchsize = 1

def Matmul(input_node, in_dim, out_dim, name, act_func=None):
    w = ng.variable(dtype=weight_dtype, shape=(out_dim, in_dim), name=f"{name}.w")
    b = ng.variable(dtype=weight_dtype, shape=(out_dim,), name=f"{name}.b")
    s = ng.variable(dtype=weight_dtype, shape=(out_dim,), name=f"{name}.s")
    rshift = ng.variable(dtype=weight_dtype, shape=(1,), name=f"{name}.rshift")
    out = ng.matmul(input_node, w, b, s, rshift_out=rshift, 
                    transposed_a=False, transposed_b=True, act_func=act_func,
                    dtype=act_dtype, sum_dtype=sum_dtype, name=name)
    return out

input_layer_name = "l0"
output_layer_name = "l3"
l0 = ng.placeholder(dtype=act_dtype, shape=(batchsize, 784), name=input_layer_name)
l1 = Matmul(l0, 784, 100, "l1", ng.relu)
l2 = Matmul(l1, 100, 100, "l2", ng.relu)
l3 = Matmul(l2, 100, 10, output_layer_name)
model = l3

generate_ip = sys.argv[-1] == "-ip"
if generate_ip:
    par_ich = 2
    par_och = 2
    axi_datawidth = 32

    l1.attribute(par_ich=par_ich, par_och=par_och)
    l2.attribute(par_ich=par_ich, par_och=par_och)
    l3.attribute(par_ich=par_ich, par_och=par_och)

    ng.to_ipxact([model], 'mlp', config={'maxi_datawidth': axi_datawidth})
else:
    from trainer import Trainer

    trainer = Trainer(model)
    trainer.load_address_info("address_info.txt")
    trainer.load_params("params.npz")

    trainer.train(epoch=10)
