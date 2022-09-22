# Adding MDLA to pytorch backend

This folder contains example implementation of mdla backend for pytorch [torchdynamo](https://github.com/pytorch/torchdynamo).

## Install

Install [torchdynamo](https://github.com/pytorch/torchdynamo)
```
pip3 install torchdynamo
```

## Build and Test tmdla

Run:
```
python3 setup.py install
```

## Run examples

Run torch.trace graph with mdla

```
python3 test.py
```

Run torchdynamo with mdla

```
python3 test_dynamo.py
```

Run a torchvision model with mdla

```
python3 test_model.py alexnet
```

## Torchscript MDLA

For running MDLA with torchscript refer to [here](torchscript/README.md)
