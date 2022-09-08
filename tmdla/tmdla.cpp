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
int th_debug = 0;

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

// aten:: operations that is supported and can be combined together into mdla::CompilationGroup
bool canHandle(const torch::jit::Node *node)
{
    switch (node->kind())
    {
    case prim::Constant:
    case prim::ListConstruct:
    case aten::conv1d:
    case aten::conv2d:
    case aten::_convolution:
    case aten::linear:
    case aten::view:
    case aten::batch_norm:
    case aten::cat:
    case aten::upsample_nearest2d:
        //        case aten::matmul:
        //        case aten::mm:
        //        case aten::addmm:
    case aten::mul:
    case aten::add:
    case aten::sub:
    case aten::relu:
    case aten::relu_:
    case aten::tanh:
    case aten::sigmoid:
    // case aten::avg_pool2d:
    case aten::max_pool2d:
        return true;
    default:
        return false;
    }
    return false;
}

void *prev_cmem = NULL;//get the device init from prev context obj
uint64_t laddr_off = 0;// address to combine models in memory

std::string cmd_options = "";
/*!
set MDLA compile options https://github.com/micronDLA/SDK/blob/master/docs/Codes.md
*/
void tmdla_options(std::string opt)
{
    cmd_options = opt;
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

    // create net
    thnets::THInit();
    thnets::network *net = thnets::create_network(g_size);
    net->nelem = 0;
    ninputnames = 0;
    torch::Tensor in_tensor;
    bool ret, first_node = true;
    // get layers
    int n = 0;
    for (auto node : graph.nodes())
    {
        int num_input = 1;
        if (canHandle(node) && node->kind() != prim::Constant && node->kind() != prim::ListConstruct)
        {
            bool added_module = true;
            if (first_node) // TODO: find all nodes that takes input
            {
                ret = find_tensor(node->input(0), &in_tensor, value_to_tensor);
                assert(ret);
            }

            if (node->kind() == aten::conv2d || node->kind() == aten::_convolution || node->kind() == aten::conv1d)
            {
                // at::conv2d(input, weight, bias, stride, padding, dilation, groups);
                // at::_convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled);

                at::Tensor kk, bb;
                ret = find_tensor(node->input(1), &kk, value_to_tensor);
                assert(ret);
                assert(kk.has_storage());
                float *weight = (float *)kk.data_ptr();
                ret = find_tensor(node->input(2), &bb, value_to_tensor);
                float *bias = NULL;
                if (ret)
                {
                    bias = (float *)bb.data_ptr();
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

                if (transpose)
                {
                    int tt = inp; // swap inp and outp
                    inp = outp;
                    outp = tt;
                    if (th_debug > 1)
                        printf("trans_spconv_%d_%d_%dx%ds%dx%dp%dx%ddl%dx%dgrp%d\n", inp, outp, kW, kH, dW, dH, pW, pH, dlW, dlH, group);
                    thload_TransposedConv2d(net->modules + n, weight, bias, inp, outp, kW, kH, pW, pH, dW, dH, opW, opH, group);
                }
                else
                {
                    if (th_debug > 1)
                        printf("spconv_%d_%d_%dx%ds%dx%dp%dx%ddl%dx%dgrp%d\n", inp, outp, kW, kH, dW, dH, pW, pH, dlW, dlH, group);
                    thload_Conv2d(net->modules + n, weight, bias, inp, outp, kW, kH, pW, pH, dW, dH, dlW, dlH, group);
                }
            }
            else if (node->kind() == aten::linear)
            {
                at::Tensor kk, bb;
                ret = find_tensor(node->input(1), &kk, value_to_tensor);
                assert(ret);
                assert(kk.has_storage());
                float *weight = (float *)kk.data_ptr();
                ret = find_tensor(node->input(2), &bb, value_to_tensor);
                float *bias = NULL;
                if (ret)
                {
                    bias = (float *)bb.data_ptr();
                }
                int i = 0, o = 0;
                o = kk.sizes()[0];
                i = kk.sizes()[1];
                if (th_debug > 1)
                    printf("linear_%dx%d\n", i, o);
                thload_Linear(net->modules + n, weight, bias, i, o);
            }
            else if (node->kind() == aten::cat)
            {
                // c10::List<at::Tensor> lten = get_listtensor(&value_to_ivalue[node->input(0)]);
                Node *nn = node->input(0)->node();
                int axis = get_const_int(node->input(1));
                num_input = nn->inputs().size();
                for (unsigned x = 0; x < nn->inputs().size(); x++)
                {
                    net->modules[n].inputs.push_back(get_node_inputnames(nn, net, x));
                    net->modules[n].all_inputs.push_back(0);
                }
                if (th_debug > 1)
                    printf("concat\n");
                thload_Concat(net->modules + n, axis);
            }
            else if (node->kind() == aten::upsample_nearest2d)
            {
                int h_scale = get_const_intlist(node->input(1))[0];
                int w_scale = get_const_intlist(node->input(1))[1];
                if (th_debug > 1)
                    printf("upsample_%dx%d\n", w_scale, h_scale);
                thload_Upsample(net->modules + n, w_scale, h_scale);
            }
            else if (node->kind() == aten::add)
            {
                num_input = 2;
                net->modules[n].inputs.push_back(get_node_inputnames(node, net, 0));
                net->modules[n].all_inputs.push_back(0);
                net->modules[n].inputs.push_back(get_node_inputnames(node, net, 1));
                net->modules[n].all_inputs.push_back(0);
                if (th_debug > 1)
                    printf("add\n");
                thload_Add(net->modules + n);
            }
            else if (node->kind() == aten::sub)
            {
                num_input = 2;
                net->modules[n].inputs.push_back(get_node_inputnames(node, net, 0));
                net->modules[n].all_inputs.push_back(0);
                net->modules[n].inputs.push_back(get_node_inputnames(node, net, 1));
                net->modules[n].all_inputs.push_back(0);
                if (th_debug > 1)
                    printf("sub\n");
                thload_Sub(net->modules + n);
            }
            else if (node->kind() == aten::mul)
            {
                num_input = 2;
                net->modules[n].inputs.push_back(get_node_inputnames(node, net, 0));
                net->modules[n].all_inputs.push_back(0);
                net->modules[n].inputs.push_back(get_node_inputnames(node, net, 1));
                net->modules[n].all_inputs.push_back(0);
                if (th_debug > 1)
                    printf("mul\n");
                thload_BatchNorm(net->modules + n, NULL, NULL, NULL, NULL, 0, 1);
            }
            else if (node->kind() == aten::batch_norm)
            {
                // aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled)
                float *weight = NULL;
                int s = 0;
                at::Tensor bb;
                ret = find_tensor(node->input(1), &bb, value_to_tensor);
                if (ret)
                {
                    s = bb.sizes()[0];
                    weight = (float *)bb.data_ptr();
                }
                float *bias = NULL;
                ret = find_tensor(node->input(2), &bb, value_to_tensor);
                if (ret)
                {
                    if (s != bb.sizes()[0])
                        s = bb.sizes()[0];
                    bias = (float *)bb.data_ptr();
                }
                float *run_mean = NULL;
                ret = find_tensor(node->input(3), &bb, value_to_tensor);
                if (ret)
                {
                    if (s != bb.sizes()[0])
                        s = bb.sizes()[0];
                    run_mean = (float *)bb.data_ptr();
                }
                float *run_var = NULL;
                ret = find_tensor(node->input(3), &bb, value_to_tensor);
                if (ret)
                {
                    if (s != bb.sizes()[0])
                        s = bb.sizes()[0];
                    run_var = (float *)bb.data_ptr();
                }
                int eps = get_const_double(node->input(7));
                if (th_debug > 1)
                    printf("batchnorm\n");
                thload_BatchNorm(net->modules + n, weight, bias, run_mean, run_var, eps, s);
            }
            else if (node->kind() == aten::relu || node->kind() == aten::relu_)
            {
                if (th_debug > 1)
                    printf("relu\n");
                thload_Threshold(net->modules + n);
            }
            else if (node->kind() == aten::tanh)
                thload_Tanh(net->modules + n);
            else if (node->kind() == aten::sigmoid)
                thload_Sigmoid(net->modules + n);
            else if (node->kind() == aten::view)
                thload_View(net->modules + n);

            else if (node->kind() == aten::max_pool2d || node->kind() == aten::avg_pool2d)
            {
                // aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False)
                // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None)
                int kW, kH, dW, dH, pW, pH, dlW, dlH, ceil;
                kW = get_const_intlist(node->input(1))[0];
                kH = get_const_intlist(node->input(1))[1];
                if (!get_const_intlist(node->input(2)).empty())
                {
                    dW = get_const_intlist(node->input(2))[0];
                    dH = get_const_intlist(node->input(2))[1];
                }
                else
                {
                    dW = kW;
                    dH = kH;
                }
                pW = get_const_intlist(node->input(3))[0];
                pH = get_const_intlist(node->input(3))[1];
                if (node->kind() == aten::max_pool2d)
                {
                    dlW = get_const_intlist(node->input(4))[0];
                    dlH = get_const_intlist(node->input(4))[1];
                    ceil = get_const_bool(node->input(5));
                    if (th_debug > 1)
                        printf("maxpool_%dx%ds%dx%dp%dx%d\n", kW, kH, dW, dH, pW, pH);
                    thload_Maxpool2d(net->modules + n, kW, kH, pW, pH, dW, dH, dlW, dlH, ceil);
                }
                else
                {
                    ceil = get_const_bool(node->input(4));
                    if (th_debug > 1)
                        printf("avgpool_%dx%ds%dx%dp%dx%d\n", kW, kH, dW, dH, pW, pH);
                    thload_Avgpool2d(net->modules + n, kW, kH, pW, pH, dW, dH, ceil);
                }
            }
            else
            {
                added_module = false;
            }
            // TODO: get name of input and output
            // connect layers with input output ids
            if (added_module)
            {
                std::string in_id = std::to_string(node->input(0)->unique());
                net->modules[n].inputs.push_back(get_node_inputnames(node, net, 0));
                net->modules[n].all_inputs.push_back(0);
                for (int j = 0; j < (int) node->outputs().size(); j++)
                {
                    std::string outstr = std::to_string(node->output(j)->unique());
                    net->modules[n].outputs.push_back(thnets::THNTensor_new(thnets::DT_FLOAT, outstr.c_str()));
                }
                net->modules[n].net = net;
                net->nelem = ++n;
                first_node = false;
            }
        }
    } // all graph.nodes

    for (size_t i = 1; i < graph.inputs().size(); ++i) // weights, bias may come from input
    {
        auto value_input = graph.inputs()[i];
        value_to_tensor[value_input] = tensors[i - 1];
    }
    // get input shapes
    // TODO: support list of inputs
    char image[100], sbuf[10];
    unsigned ninputs = 1;
    int inP = 1, inSz = 1;
    { // get input size
        inP = (in_tensor.sizes().size() == 1) ? in_tensor.sizes()[0] : in_tensor.sizes()[1];
        sprintf(image, "1x%d", inP);
        inSz = inP;
        for(int j = 2; j < (int) in_tensor.sizes().size(); j++){
            sprintf(sbuf, "x%d", (int) in_tensor.sizes()[j]);
            inSz *= in_tensor.sizes()[j];
            strcat(image, sbuf);
        }
    }
    thnets::thnetwork_add_input(net, std::to_string(0).c_str());
    for (unsigned n = 0; n < graph.outputs().size(); n++)
    {
        for (int i = 0; i < net->nelem; i++)
        {
            for (size_t j = 0; j < net->modules[i].outputs.size(); ++j)
            {
                bool scm = strcmp(net->modules[i].outputs[j]->name.c_str(), std::to_string(graph.outputs()[n]->unique()).c_str());
                if (scm == 0)
                {
                    net->modules[i].outidx[j] = n + 1;
                    thnets::thnetwork_add_output(net, std::to_string(n + 1).c_str());
                    break;
                }
            }
        }
    }

    //compile MDLA
    void *cmem;
    Compiled_info *cinfo;

    cmem = ie_create();
    if (prev_cmem!=NULL)
    { // combine the code in main memory and use same pico obj
        char s[100];
        sprintf(s, "%ld", laddr_off);
        ie_setflag(cmem, "addr_off", s);//set addroff for new cmem to be combined with prev_cmem
    }
    cinfo = new Compiled_info(cmem);
    if (th_debug > 1)
        printf("OPTIONS: %s\n", cmd_options.c_str());

    std::string sp = " ";
    std::string cur_str = cmd_options;
    std::string prev_str = "";
    size_t pos = 0;
    cur_str.append(" ");
    while ((pos = cur_str.find(sp)) != std::string::npos) 
    {
        if(prev_str == "-d")
        {
            ie_setflag(cmem, "debug", cur_str.substr(0, pos).c_str());
            prev_str = "";
        }
        else if(prev_str == "-o")
        {
            ie_setflag(cmem, "options", cur_str.substr(0, pos).c_str());
            prev_str = "";
        }
        else if(prev_str == "-c")
        {
            ie_setflag(cmem, "nclusters", cur_str.substr(0, pos).c_str());
            prev_str = "";
        }
        else
            prev_str = cur_str.substr(0, pos);
        cur_str.erase(0, pos + sp.length());
    }

    unsigned *noutdims;
    unsigned noutputs = 0;
    uint64_t **outshapes;
    if (th_debug > 1)
        printf("compile with image size: %s\n", image);
    // pass THNETWORK to thnets2lst to create lst
    ext_thnets2lst(cmem, net, image, 1);
    // ie_compile: skip onnx parser if already lst exist and modelpath="$keeplst"
    cmem = ie_compile(cmem, "$keeplst", 0, image, &noutputs, &noutdims, &outshapes, prev_cmem);
    ie_getinfo(cmem, "addr_off", &laddr_off, sizeof(laddr_off));//save cmem and addr_off for next compiled subgraph
    prev_cmem = cmem;
    cinfo->laddr_off = laddr_off;
    cinfo->input_elements[0] = inSz;
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
    m.def("tmdla_options", &tmdla_options, "tmdla_options");
}
