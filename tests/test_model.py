import torch
import torchdynamo
from tmdla import to_mdla, print_err, tmdla_options
from torchvision import models
from argparse import ArgumentParser

parser = ArgumentParser(
    description="Test run torchvision model on pytorch_mdla")
_ = parser.add_argument
_('model', type=str, default='', help='Model name')
args = parser.parse_args()

example_inputs = torch.rand(1, 3, 224, 224)
nn_module = getattr(models, args.model)(pretrained=True).eval()

tmdla_options("-d bw")

with torchdynamo.optimize(to_mdla):  # run using MDLA
    yy = nn_module(example_inputs)

y = nn_module(example_inputs)
print_err(yy.detach().numpy(), y.detach().numpy())
