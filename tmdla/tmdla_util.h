// Copyright (c) 2019 Micron Technology, Inc. All Rights Reserved. This source code contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc.

#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include "api.h"

#define MAX_INPUTS 20
#define MAX_OUTPUTS 256

using namespace torch::jit;

class Compiled_info
{
public:
    void *cmem;
    uint64_t input_elements[MAX_INPUTS];
    unsigned ninputs;

    uint64_t output_elements[MAX_INPUTS];
    unsigned noutputs;
    unsigned *noutdims;
    uint64_t **outshapes;

    Compiled_info(void *cm) : cmem(cm){};
    ~Compiled_info()
    {
        ie_free(cmem);
    };
};

c10::List<int64_t> get_const_intlist(Value *input);

int64_t get_const_int(Value *input);

double get_const_double(Value *input);

bool get_const_bool(Value *input);

at::Tensor get_tensor(IValue *input);

c10::List<at::Tensor> get_listtensor(IValue *input);

bool find_tensor(Value *input, at::Tensor *tensor, std::unordered_map<Value *, torch::Tensor> value_to_tensor);
