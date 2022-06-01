# License
Copyright (c) 2019 Micron Technology, Inc. All Rights Reserved. This source code contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc.

# Adding MDLA to pytorch backend

This folder contains example implementation of mdla backend for pytorch with [functorch](https://github.com/pytorch/functorch) and [torchdynamo](https://github.com/pytorch/torchdynamo).

## Install

Install [functorch](https://github.com/pytorch/functorch) and [torchdynamo](https://github.com/pytorch/torchdynamo)

## Build and Test tmdla

Add `api.h`, `thvector.h`, `thnets.h` and `thnets.def` into this folder

Then run:
```
python3 setup.py install
```

## Run examples

Run torch.trace graph with mdla

```
python3 test.py
```

Run functorch with mdla

```
python3 test_functorch.py
```

Run torchdynamo with mdla

```
python3 test_dynamo.py
```

## Torchscript MDLA

For running MDLA with torchscript refer to [here](torchscript/README.md)
