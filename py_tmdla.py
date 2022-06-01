import torch
import torch.utils
import tmdla
import numpy as np

def to_mdla(fx_module, example_inputs):

    torchscript_trace = torch.jit.trace(fx_module, example_inputs)
    torchscript_trace = torch.jit.freeze(torchscript_trace.eval())
#     lltm_cpp.tensort_debug(torchscript_trace.graph)
    linput = [i for i in example_inputs]
    v = tmdla.tmdla_compile(torchscript_trace.graph, linput)

    def exec_mdla(*args):
        outs = tmdla.tmdla_run(v, args[-1])
        return outs

    return exec_mdla

def print_err(result, result_pyt):
    error_mean = (np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
    error_max = (np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
    print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))