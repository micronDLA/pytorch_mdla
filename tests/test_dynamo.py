import torch
import torchdynamo
from tmdla import to_mdla, print_err

class modelC(torch.nn.Module):
    def __init__(self):
        super(modelC, self).__init__()
        self.op = torch.nn.Conv2d(64, 64, 1, 1)
        self.re = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=1)
        self.op.weight.data = torch.ones(64, 64, 1, 1)*4

    def forward(self, x):
        y = self.op(x)
        y = self.re(y)
        y = self.soft(y)
        return y

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.op = modelC()
        self.op2 = modelC()

    def forward(self, x):
        y = self.op(x)
        y1 = self.op2(y)
        return y1

example_inputs = torch.rand(1, 64, 1, 1)
nn_module = model()

with torchdynamo.optimize(to_mdla): #run using MDLA
    yy = nn_module(example_inputs)

y = nn_module(example_inputs)
print_err(yy.detach().numpy(), y.detach().numpy())
