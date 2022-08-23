
import torch
import torch.utils
import torch.nn.functional as F
import torch.fx as fx
import tmdla
from tmdla import print_err

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


nn_module = model()

example_inputs = torch.ones(1, INP, 1, 1)*0.1
torchscript_trace = torch.jit.trace(nn_module, example_inputs)
torchscript_trace = torch.jit.freeze(torchscript_trace.eval())

v = tmdla._c.tmdla_compile(torchscript_trace.graph, [example_inputs])
yy = tmdla._c.tmdla_run(v, example_inputs)

print(yy.shape)

y = nn_module(example_inputs)
print_err(yy.detach().numpy(), y.detach().numpy())
