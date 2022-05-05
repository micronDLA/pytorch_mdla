# Copyright (c) 2019 Micron Technology, Inc. All Rights Reserved. This source code contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc.


import numpy as np
import torch
import torch.utils
import torch.nn.functional as F
import torch.fx as fx
import tmdla

INP = 64
OUTP = 64


class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        inP = INP
        outP = OUTP
        self.op = torch.nn.Conv2d(inP, outP, 1, 1)
        for op in range(outP):
            for ip in range(inP):
                self.op.weight.data[op, ip, 0, 0] = 0.1
        for op in range(outP):
            self.op.bias.data[op] = 0.0

    def forward(self, x):
        y = self.op(x)
        return y


my_module = model()

example_inputs = torch.ones(1, INP, 1, 1)*0.1
torchscript_trace = torch.jit.trace(my_module, example_inputs)
torchscript_trace = torch.jit.freeze(torchscript_trace.eval())

v = tmdla.tmdla_compile(torchscript_trace.graph, [example_inputs])
t = tmdla.tmdla_run(v, example_inputs)
print(t)
print(t.shape)
