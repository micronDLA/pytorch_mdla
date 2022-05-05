# License
Copyright (c) 2019 Micron Technology, Inc. All Rights Reserved. This source code contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc.

# Adding MDLA to pytorch backend using functorch

This folder contains example implementation of mdla backend for pytorch with functorch.

## Install


## Build and Test tmdla

Add `api.h`, `thvector.h`, `thnets.h` and `thnets.def` into this folder

Then run:
```
python3 setup.py install
```

Then install functorch from [here](https://github.com/pytorch/functorch)

## Run examples

Run torch.trace graph with mdla

```
python3 test.py
```

Run functorch with mdla

```
python3 test_functorch.py
```


