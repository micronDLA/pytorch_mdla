import torch
import torch.utils
import tmdla
import numpy as np

import torch.fx as fx
from torch.fx import Node
from torch.fx.passes.split_module import split_module
from functools import partial
from typing import List

class MDLA_parts:

    def __init__(self):
        self.partition_counter = 0
        self.prev_type = ''

    def mdla_partition(self, node: Node):
        tag = 'mdla' if 'mdla' in node.name else 'cpu'
        if tag != self.prev_type:
            self.partition_counter += 1
        self.prev_type = tag
        partition = tag + '_' + str(self.partition_counter) #add mdla to subgraph name to run on mdla
        return partition

def tmdla_run(tensor_in, mod=None):
    out = tmdla._c.tmdla_run(mod, tensor_in)
    return out

def to_mdla(fx_trace: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    compile pytorch model to run on MDLA using torchdynamo
    Args:
        fx_trace: torch.fx.GraphModule
        example_inputs: list of model input tensors
    """
    # label layers supported in mdla
    for nd in fx_trace.graph.nodes:
        if nd.op == 'call_module':
            mod = fx_trace.get_submodule(nd.target)
            if isinstance(mod, torch.nn.ReLU) or isinstance(mod, torch.nn.Conv2d):
                nd.name += "_mdla" #add mdla tag to name

    # split into subgraphs to run on mdla using node name
    mp = MDLA_parts()
    module_with_submodules = split_module(fx_trace, fx_trace, mp.mdla_partition)

    exec_graph = [] # list of compiled functions to run
    xx = example_inputs[0]
    for n in module_with_submodules.graph.nodes:
        if n.op == 'call_module':
            a = module_with_submodules.get_submodule(n.target)
            if 'mdla' in n.name: # run on mdla
                ts_trace = torch.jit.trace(a, xx)
                ts_trace = torch.jit.freeze(ts_trace.eval())
                v = tmdla._c.tmdla_compile(ts_trace.graph, [xx])
                fun = partial(tmdla_run, mod=v)
                exec_graph.append(fun)
            else: # fallback to pytorch run
                exec_graph.append(a)
            xx = a(xx)

    def exec_mdla(*args):
        outs = None
        for fun in exec_graph:
            if outs is None:
                outs = fun(args[-1])
            else:
                outs = fun(outs)
        return [outs]

    return exec_mdla  # return a python callable

def print_err(result, result_pyt):
    error_mean = (np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
    error_max = (np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
    print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))
