import nngen as ng
from npng import PlaceHolder, Matmul
from common import input_layer_name, output_layer_name

if __name__ == '__main__':
    l0 = PlaceHolder(784, input_layer_name)
    l1 = Matmul(l0, 784, 100, "l1")
    l2 = Matmul(l1, 100, 100, "l2")
    l3 = Matmul(l2, 100, 10, output_layer_name, act=False)

    par_ich = 2
    par_och = 2
    axi_datawidth = 32

    l1.out_ng.attribute(par_ich=par_ich, par_och=par_och)
    l2.out_ng.attribute(par_ich=par_ich, par_och=par_och)
    l3.out_ng.attribute(par_ich=par_ich, par_och=par_och)

    ng.to_ipxact([l3.out_ng], 'mlp', config={'maxi_datawidth': axi_datawidth})
