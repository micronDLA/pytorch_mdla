import torch
import torchdynamo
from py_tmdla import to_mdla, print_err

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

example_inputs = torch.ones(1, INP, 1, 1)*0.1
nn_module = model()

with torchdynamo.optimize(to_mdla):
    yy = nn_module(example_inputs)

print(yy.shape)

y = nn_module(example_inputs)
print_err(yy.detach().numpy(), y.detach().numpy())
