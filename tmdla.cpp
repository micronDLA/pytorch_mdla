// Copyright (c) 2019 Micron Technology, Inc. All Rights Reserved. This source code contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc.

#include "tmdla_util.h"
#include <pybind11/pybind11.h>

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include "thnets.h"

namespace py = pybind11;
using namespace torch::jit;

static std::string inputnames[MAX_INPUTS];
static int ninputnames;

static int get_node_inputnames(Node *node, thnets::network *net, int n)
{
    std::string in_id = std::to_string(node->input(n)->unique());
    int mod_idx = thnets::getoutput_c(net, in_id.c_str()); // index of module that created this input
    if (mod_idx == -1)                                     // there isn't a module that created that identifier. It must be model input
    {
        int k = -1;
        for (int x = 0; x < ninputnames; x++)
            if (inputnames[x] == in_id)
            {
                k = x;
                break;
            }
        if (k == -1)
        {
            if (ninputnames == MAX_INPUTS)
                thnets::THError("Maximum number of inputs (%d) exceeded\n", MAX_INPUTS);
            else
            {
                inputnames[ninputnames] = in_id;
                k = ninputnames++;
            }
        }
        mod_idx = -1 - k; // Inputs are numbered -1, -2, -3...
    }
    return mod_idx;
}

/*!
compile torchscript graph <torch.Graph> to run on MDLA
    @param graph: torch.Graph
    @param tensors: List of torch.Tensor
*/
void *tmdla_compile(Graph &graph, std::vector<torch::Tensor> &tensors)
{
    std::unordered_map<Value *, torch::Tensor> value_to_tensor;
    for (size_t i = 1; i < graph.inputs().size(); ++i) // weights, bias may come from input
    {
        auto value_input = graph.inputs()[i];
        value_to_tensor[value_input] = tensors[i - 1];
    }

    int g_size = 0;
    for (auto node : graph.nodes())
        g_size++;

    void *cmem;
    cmem = ie_create();

    // create net
    thnets::THInit();
    thnets::network *net = thnets::create_network(g_size);
    net->nelem = 0;
    ninputnames = 0;
    torch::Tensor in_tensor;
    bool first_node = true;
    // get layers
    int n = 0;
    for (auto node : graph.nodes())
    {
        int num_input = 1;
        if (node->kind() != prim::Constant &&
            node->kind() != prim::ListConstruct &&
            (node->kind() == aten::conv2d || node->kind() == aten::_convolution || node->kind() == aten::conv1d))
        {
            // at::conv2d(input, weight, bias, stride, padding, dilation, groups);
            // at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);
            bool ret;
            if (first_node) // TODO: find all nodes that takes input
            {
                ret = find_tensor(node->input(0), &in_tensor, value_to_tensor);
                assert(ret);
            }

            at::Tensor kk, bb;
            ret = find_tensor(node->input(1), &kk, value_to_tensor);
            assert(ret);
            assert(kk.has_storage());
            float *weight = (float *)kk.data_ptr();
            std::cout << "Found kk find_tensor " << kk.sizes() << std::endl;
            ret = find_tensor(node->input(2), &bb, value_to_tensor);
            float *bias = NULL;
            if (ret)
            {
                bias = (float *)bb.data_ptr();
                std::cout << "Found bb find_tensor " << bb.sizes() << std::endl;
            }
            int outp, inp, kW, kH = 1, dW, dH = 1, pW, pH = 0, dlW, dlH = 1, opW = 0, opH = 0, group;
            bool transpose = false;
            if (node->kind() == aten::conv1d)
            { // conv1d
                outp = kk.sizes()[0];
                inp = kk.sizes()[1];
                kW = kk.sizes()[2];
                dW = get_const_intlist(node->input(3))[0];
                pW = get_const_intlist(node->input(4))[0];
                dlW = get_const_intlist(node->input(5))[0];
                group = get_const_int(node->input(6));
            }
            else
            { // conv2d
                outp = kk.sizes()[0];
                inp = kk.sizes()[1];
                kW = kk.sizes()[3];
                kH = kk.sizes()[2];
                if (node->kind() == aten::_convolution)
                {
                    dH = dW = get_const_intlist(node->input(3))[0];
                    pH = pW = get_const_intlist(node->input(4))[0];
                    dlH = dlW = get_const_intlist(node->input(5))[0];
                    opH = opW = get_const_intlist(node->input(7))[0];
                    transpose = get_const_bool(node->input(6));
                    group = get_const_int(node->input(8));
                }
                else
                {
                    dW = get_const_intlist(node->input(3))[0];
                    dH = get_const_intlist(node->input(3))[1];
                    pW = get_const_intlist(node->input(4))[0];
                    pH = get_const_intlist(node->input(4))[1];
                    dlW = get_const_intlist(node->input(5))[0];
                    dlH = get_const_intlist(node->input(5))[1];
                    group = get_const_int(node->input(6));
                }
            }
            printf("spconv_%dx%dx%ds%dx%dp%dx%ddl%dx%dgrp%d\n", outp, kW, kH, dW, dH, pW, pH, dlW, dlH, group);
            if (transpose)
            {
                int tt = inp; // swap inp and outp
                inp = outp;
                outp = tt;
                thload_TransposedConv2d(net->modules + n, weight, bias, inp, outp, kW, kH, pW, pH, dW, dH, opW, opH, group);
            }
            else
                thload_Conv2d(net->modules + n, weight, bias, inp, outp, kW, kH, pW, pH, dW, dH, dlW, dlH, group);

            // TODO: get name of input and output
            // connect layers with input output ids
            if (num_input == 1)
            { // one input layers
                net->modules[n].inputs.push_back(get_node_inputnames(node, net, 0));
                net->modules[n].all_inputs.push_back(0);
            }

            std::string outstr = std::to_string(node->output(0)->unique());
            net->modules[n].outputs.push_back(thnets::THNTensor_new(thnets::DT_FLOAT, outstr.c_str()));
            net->modules[n].outidx[0] = 1; // TODO: set this correctly
            net->nelem = ++n;
            first_node = false;
        }
    }
    // get input shapes
    // TODO: support list of inputs
    char image[100];
    int inW = 1, inH = 1, inP = 1, inZ = 1;
    unsigned ninputs = 1;
    { // get input size
        inP = in_tensor.sizes()[1];
        if (in_tensor.sizes().size() == 4)
        { // WxHxPxB
            inH = in_tensor.sizes()[2];
            inW = in_tensor.sizes()[3];
            sprintf(image, "1x%dx%dx%d", inP, inH, inW);
        }
        else if (in_tensor.sizes().size() == 5)
        { // WxHxZxPxB
            inZ = in_tensor.sizes()[2];
            inH = in_tensor.sizes()[3];
            inW = in_tensor.sizes()[4];
            sprintf(image, "1x%dx%dx%dx%d", inP, inZ, inH, inW);
        }
        else if (in_tensor.sizes().size() == 1)
        { // P
            inP = in_tensor.sizes()[0];
            sprintf(image, "1x%dx%dx%d", inP, 1, 1);
        }
        else
        { // WxPxB
            inW = in_tensor.sizes()[2];
            sprintf(image, "1x%dx%dx%d", inP, 1, inW);
        }
    }
    thnets::thnetwork_add_input(net, std::to_string(0).c_str());
    thnets::thnetwork_add_output(net, std::to_string(1).c_str());
    // TODO: pass debug as param
    std::string debug = "bw";
    std::string options = "";
    std::string clusters_ = "1";
    if (!debug.empty())
        ie_setflag(cmem, "debug", debug.c_str());
    if (!options.empty())
        ie_setflag(cmem, "options", options.c_str());
    if (clusters_ != "1")
    {
        ie_setflag(cmem, "nclusters", clusters_.c_str());
    }
    unsigned *noutdims;
    unsigned noutputs = 0;
    uint64_t **outshapes;
    // pass THNETWORK to thnets2lst to create lst
    ext_thnets2lst(cmem, net, image, 1);
    // ie_compile: skip onnx parser if already lst exist and modelpath="$keeplst"
    cmem = ie_compile(cmem, "$keeplst", 0, image, &noutputs, &noutdims, &outshapes, 0);
    Compiled_info *cinfo = new Compiled_info(cmem);

    cinfo->input_elements[0] = inP * inH * inW;
    cinfo->ninputs = ninputs;

    cinfo->noutputs = noutputs;
    cinfo->noutdims = noutdims;
    cinfo->outshapes = outshapes;
    for (unsigned i = 0; i < noutputs; i++)
    {
        cinfo->output_elements[i] = std::accumulate(outshapes[i], outshapes[i] + noutdims[i], 1, std::multiplies<int>());
    }
    return (void *)cinfo;
}

/*!
run a MDLA compiled graph
    @param cinf: MDLA compiled graph <Compiled_info> created using tmdla_compile
    @param tensor: input tensor
*/
torch::Tensor tmdla_run(void *cinf, torch::Tensor &tensor)
{
    Compiled_info *cinfo = (Compiled_info *)cinf;
    float *input_t = (float *)tensor.data_ptr();
    float *output[MAX_OUTPUTS] = {0};
    output[0] = (float *)malloc(cinfo->output_elements[0] * sizeof(float));

    int err = 0;
    err = ie_run(cinfo->cmem, &input_t, cinfo->input_elements, cinfo->ninputs, output, cinfo->output_elements, cinfo->noutputs);
    if (err)
        fprintf(stderr, "ie_run ERROR %d\n", err);

    // TODO: support multiple outputs
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    std::vector<int64_t> shape(cinfo->outshapes[0], cinfo->outshapes[0] + cinfo->noutdims[0]);
    torch::Tensor out_tensor = torch::from_blob(output[0], shape, options);
    return out_tensor;
}

/*!
pybind: create tmdla module with tmdla_compile and tmdla_run functions
*/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("tmdla_compile", &tmdla_compile, "tmdla_compile");
    m.def("tmdla_run", &tmdla_run, "tmdla_run");
}
