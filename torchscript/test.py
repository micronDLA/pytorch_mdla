import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional  as F
import time
import build.torchMDLA as tmdla




class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(3, 32, 3, 1)

    def forward(self, x):
        y = self.conv1(x)
        w = self.conv2(x)
        z = torch.cat([y, w], 1)
        return z

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.upsample = nn.Upsample(size=8)

    def forward(self, x):
        y = self.conv1(x)
        z = self.upsample(y)
        return z

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        #self.conv1 = nn.Conv2d(3, 32, 3, 1, bias=False)
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(3, 32, 3, 1)
        self.bn = nn.BatchNorm2d(32)
        self.lin = nn.Linear(128, 64)
        self.tconv1 = nn.ConvTranspose2d(3, 32, 3, 1)
        #outP = 32
        #inP = 3
        #for op in range(outP):
        #    for ip in range(inP):
        #        for y in range(3):
        #            for x in range(3):
        #                self.conv1.weight.data[op,ip,y,x] = 0.01
        #for op in range(outP):
        #    self.conv1.bias.data[op] = 1

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))
        x = x.view(-1)
        x = self.lin(x)
        return x

def acc_check(result, result_pyt):
    error_mean=(np.absolute(result-result_pyt).mean()/np.absolute(result_pyt).max())*100.0
    error_max=(np.absolute(result-result_pyt).max()/np.absolute(result_pyt).max())*100.0
    print('\x1b[32mMean/max error compared to pytorch are {:.3f}/{:.3f} %\x1b[0m'.format(error_mean, error_max))

def test1(model):
    x = torch.rand(1, 3, 6, 6)
    with torch.no_grad():
        tmdla.enable()
        trace_jit = torch.jit.trace(model, x, check_trace=False, check_tolerance=2)
        trace_jit(x) # run optimize and have execution_plan
        output = trace_jit(x) # run with mdla
        # print(torch.jit.last_executed_optimized_graph())
        tmdla.disable()
        #run without mdla
        output_py = model(x)
        output = output.view(-1)
        output_py = output_py.view(-1)
        acc_check(output, output_py)

def test2():
    @torch.jit.script
    def foo_jit(a, b):
      c = F.conv2d(a, b)
      c = F.relu(c)
      c = F.avg_pool2d(c, (2,2))
      c = F.max_pool2d(c, (2,2))
      return c

    def foo_jit2(a, b):
      c = F.conv2d(a, b)
      c = F.relu(c)
      c = F.avg_pool2d(c, (2,2))
      c = F.max_pool2d(c, (2,2))
      return c
    x = torch.rand(1, 16, 8, 8)
    h = torch.rand(64, 16, 3, 3)
    tmdla.enable(debug="", options="C", clusters=2)
    traced_cell = torch.jit.trace(foo_jit, (x, h), check_trace=False)
    traced_cell(x, h) # run optimize and have execution_plan
    output = traced_cell(x, h) # run with mdla
    # print(torch.jit.last_executed_optimized_graph())
    tmdla.disable()
    output_py = foo_jit2(x, h)
    output = output.view(-1)
    output_py = output_py.view(-1)
    acc_check(output, output_py)

if __name__ == "__main__":
    test1(Net1())
    test1(Net2())
    test1(Net3())
    test2()

