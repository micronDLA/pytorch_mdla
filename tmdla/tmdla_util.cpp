// Copyright (c) 2019 Micron Technology, Inc. All Rights Reserved. This source code contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc.

#include "tmdla_util.h"

using namespace torch::jit;

c10::List<int64_t> get_const_intlist(Value *input)
{
    Node *nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isIntList());
    return ivalue->toIntList();
}

int64_t get_const_int(Value *input)
{
    Node *nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isInt());
    return ivalue->toInt();
}

double get_const_double(Value *input)
{
    Node *nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isDouble());
    return ivalue->toDouble();
}

bool get_const_bool(Value *input)
{
    Node *nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isBool());
    return ivalue->toBool();
}

at::Tensor get_tensor(IValue *input)
{
    assert(input->isTensor());
    return input->toTensor();
}

c10::List<at::Tensor> get_listtensor(IValue *input)
{
    assert(input->isTensorList());
    return input->toTensorList();
}

static void print_node(Node *node)
{
    std::cout << "Running this " << node->kind().toDisplayString() << " inputs: " << node->inputs().size() << std::endl;
    std::cout << "input:";
    for (int ii = 0; ii < (int) node->inputs().size(); ii++)
    {
        std::cout << node->inputs()[ii]->unique() << ", ";
    }
    std::cout << std::endl;
    std::cout << "output: ";
    for (int ii = 0; ii < (int) node->outputs().size(); ii++)
    {
        std::cout << node->outputs()[ii]->unique() << ", ";
    }
    std::cout << std::endl;
}

bool find_tensor(Value *input, at::Tensor *tensor, std::unordered_map<Value *, torch::Tensor> value_to_tensor)
{
    if (value_to_tensor.find(input) != value_to_tensor.end())
    {
        *tensor = value_to_tensor[input];
        return true;
    }
    else
    {
        Graph *graph = input->owningGraph();
        for (auto node : graph->nodes())
        {
            if (node->kind() == prim::Constant && input == node->outputs()[0])
            {

                auto ivalue = toIValue(node->outputs()[0]);
                if (ivalue->isTensor())
                {
                    *tensor = ivalue->toTensor();
                    return true;
                }
            }
        }
    }
    return false;
}
