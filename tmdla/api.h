//#Copyright 2019 Micron Technology, Inc. All Rights Reserved. This software contains confidential information and trade secrets of Micron Technology, Inc. Use, disclosure, or reproduction is prohibited without the prior express written permission of Micron Technology, Inc
/// @file
/// @brief Micron DLA C api
#ifndef THNETS_H
#define THNETS_H

#ifdef _WIN32
#define IECOMPILER_API __declspec(dllexport)
#else
#define IECOMPILER_API
#endif

#ifndef _IE_API_H_INCLUDED_
#define _IE_API_H_INCLUDED_

static const char *microndla_version = "2022.1.0";
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __cplusplus
extern "C"
{
#endif

#define THAtomicIncrement(a) __sync_fetch_and_add(a, 1);
#define THAtomicDecrement(a) __sync_fetch_and_add(a, -1);

#ifdef INT8
    typedef int8_t SF_INT;
#else
typedef int16_t SF_INT;
#endif

    /*!
    Allow to reference externally defined functions for interface the hardware
        @param cmemo        context object, can be null
        @param cfgrd        function pointer for reading config registers from DLA
        @param cfgwr        function pointer for writing config registers to DLA
        @param readext      function pointer for reading data from external memory connected with DLA
        @param writeext     function pointer for writing data to external memory connected with DLA
        @return 0 pass, -1 fail
    */
    int IECOMPILER_API set_external_interface(void *cmemo, int64_t (*cfgrd)(uint64_t), void (*cfgwr)(uint64_t, uint64_t),
                                              void (*readext)(uint64_t, void *, uint64_t), void (*writeext)(uint64_t, void *, uint64_t));

    /*!
    Allow to reference externally defined functions for wait/sleep in hardware simulation (veloce only)
    */
    int IECOMPILER_API set_external_wait(void *cmemo, bool (*wait_ext)(int));

    /*!
    Allow to pass externally created thnets net into node list
    */
    void IECOMPILER_API ext_thnets2lst(void *cmemo, void *nett, char *image, int batch);

    /*!
    Create an Inference Engine object
    */
    void IECOMPILER_API *ie_create();

    /*
    All-in-one: Compile a network, Initialize FPGA, and Run accelerator
        @param cmemo        pointer to an Inference Engine object object.  May be null.
        @param modelpath    path to the onnx file
        @param inshapes     shape of the inputs in the form size0xsize1xsize2...; more inputs are separated by semi-colon; this parameter is optional as the shapes of the inputs can be obtained from the model file
        @param input        input data, one pointer per input
        @param output       output data, one pointer per output
        @return -1 (error), 0 (pass)
    */
    int IECOMPILER_API ie_go(void *cmemo, const char *modelpath, const char *inshapes, const float *const *input, float **output);

    /*!
    Compile a network and produce a .bin file with everything that is needed to execute in hardware.
    const float * const *input, const uint64_t *input_elements, unsigned ninputs
    Run static quantization of inputs, weight and outputs over a calibration dataset
        @param cmemo        pointer to an Inference Engine object.   May be null
        @param modelpath    path to the onnx file
        @param outbin       path to output .bin file
        @param inshapes     shape of the inputs in the form size0xsize1xsize2...; more inputs are separated by semi-colon; this parameter is optional as the shapes of the inputs can be obtained from the model file
        @param noutputs     number of returned outputs
        @param noutdims     returns a pointer to noutputs values with the dimensions of each output
        @param outshapes    returns a pointer to noutputs pointers to the shapes of each output
        @param input            pointers to the calibration dataset
        @param input_elements   size of the input in number of elements, one per input
        @param ninputs          number of inputs, must be a multiple of the inputs expected by the network
        @return context object
    */
    void IECOMPILER_API *ie_compile_vfp(void *cmemo, const char *modelpath, const char *outbin, const char *inshapes,
                                        unsigned *noutputs, unsigned **noutdims, uint64_t ***outshapes,
                                        const float *const *inputs, const uint64_t *input_elements, unsigned ninputs, void *cmemp);

    /*!
    Compile a network and produce a .bin file with everything that is needed to execute in hardware.
    If the model contains some layers that cannot be run in hardware, they will be run in software.
    In this case, ie_compile is necessary, ie_init with a previously generated bin file is not enough
        @param cmemo        pointer to an Inference Engine object.   May be null
        @param modelpath    path to the onnx file
        @param outbin       path to output .bin file
        @param inshapes     shape of the inputs in the form size0xsize1xsize2...; more inputs are separated by semi-colon; this parameter is optional as the shapes of the inputs can be obtained from the model file
        @param noutputs     number of returned outputs
        @param noutdims     returns a pointer to noutputs values with the dimensions of each output
        @param outshapes    returns a pointer to noutputs pointers to the shapes of each output
        @return context object
    */
    void IECOMPILER_API *ie_compile(void *cmemo, const char *modelpath, const char *outbin, const char *inshapes, unsigned *noutputs, unsigned **noutdims, uint64_t ***outshapes, void *cmemp);
    /*!
    Load a .bin file into the hardware and initialize it
        @param cmemo        pointer to an Inference Engine object, may be null
        @param inbin        path to .bin file generated by ie_compile
        @param outsize      output size assuming batch 1, in number of elements, one per output
        @param noutputs     returns number of outputs
        @param cmemp        copy the FPGA info to this cmem (copies pico)
        @param noutputs     number of returned outputs
        @param noutdims     returns a pointer to noutputs values with the dimensions of each output
        @param outshapes    returns a pointer to noutputs pointers to the shapes of each output
        @param cmemp        pointer to another Inference Engine object already initialized with another network, may be null
        @return context object
    */
    void IECOMPILER_API *ie_init(void *cmemo, const char *inbin, unsigned *noutputs, unsigned **noutdims, uint64_t ***outshapes, void *cmemp);

    /*!
    Run hardware
    It does the steps sequentially. putInput, compute, getResult
        @param cmemo            pointer to an Inference Engine object
        @param input            pointers to input data
        @param input_elements   size of the input in number of elements, one per input
        @param ninputs          number of inputs
        @param output           pointers to memory buffers where the output will be saved
        @param output_elements  size allocated for output in number of elements, one per output
        @param noutputs         number of outputs
        @return -1 (error), 0 (pass)
    */
    int IECOMPILER_API ie_run(void *cmemo, const float *const *input, const uint64_t *input_elements, unsigned ninputs, float **output, uint64_t *output_elements, unsigned noutputs);

    /*!
    Send input to the hardware and start Micron DLA hardware
        @param cmemo            pointer to an Inference Engine object
        @param input            pointers to input data
        @param input_elements   size of the input in number of elements, one per input
        @param ninputs          number of inputs
        @param userparam        user defined parameter useful to associate inputs and outputs
        @return -1 (error), 0 (pass)
    */
    int IECOMPILER_API ie_putinput(void *cmemo, const float *const *input, const uint64_t *input_elements, unsigned ninputs, void *userparam);

    /*!
    Get an output from the hardware. If the blockingmode flag was set then it will wait for Micron DLA hardware to finish, otherwise it will return -1
    in case the output is not ready
        @param cmemo            pointer to an Inference Engine object
        @param output           pointers to memory buffers where the output will be saved
        @param output_elements  size allocated for output in number of elements, one per output
        @param noutputs         number of outputs
        @param userparam        userparam associated to the input
        @return -1 (error), 0 (pass)
    */
    int IECOMPILER_API ie_getresult(void *cmemo, float **output, uint64_t *output_elements, unsigned noutputs, void **userparam);

    /*!
    Set flags for the compiler
        @param cmemo    pointer to an Inference Engine object
        @param name     name of the option
        @param value    value to set the option
        @return -1 (error), 0 (pass)
    */
    int IECOMPILER_API ie_setflag(void *cmemo, const char *name, const char *value);

    /*!
    Get various info about the hardware
        @param cmemo     pointer to an Inference Engine object
        @param name      name of the info to fetch
        @param value     pointer to the returned value
        @param valuesize size of the memory buffer pointed to by value
        @return -1 (error), returns the type of value returned, 0 nothing, 1 string, 2 bool, 3 int, 4 int64, 5 float
    */
    int IECOMPILER_API ie_getinfo(void *cmemo, const char *name, void *value, size_t valuesize);

    /*!
    Run software Micron DLA emulator
    This runs the model in software using the same data precision of the accelerator
        @param cmemo            pointer to an Inference Engine object
        @param input            pointers to input data
        @param input_elements   size of the input in number of elements, one per input
        @param ninputs          number of inputs
        @param output           pointers to memory buffers where the output will be saved
        @param output_elements  size allocated for output in number of elements, one per output
        @param noutputs         number of outputs
        @return -1 (error), 0 (pass)
    */
    int IECOMPILER_API ie_run_sw(void *cmemo, const float *const *input, const uint64_t *input_elements, unsigned ninputs, float **output, uint64_t *output_elements, unsigned noutputs);

    /*!
    Run the model with thnets
    args:
        @param cmemo            pointer to an Inference Engine object
        @param input            pointers to input data
        @param input_elements   size of the input in number of elements, one per input
        @param ninputs          number of inputs
        @param output           pointers to memory buffers where the output will be saved
        @param output_elements  size allocated for output in number of elements, one per output
        @param noutputs         number of outputs
        @return -1 (error), 0 (pass)
    */
    int IECOMPILER_API ie_run_thnets(void *cmemo, const float *const *input, const uint64_t *input_elements, unsigned ninputs, float **output, uint64_t *output_elements, unsigned noutputs);

    /*!
    Free FPGA instance
        @param cmemo            pointer to an Inference Engine object
    */
    void IECOMPILER_API ie_free(void *cmemo);

    /*!
    Read data from an address in shared memory.
        @param cmemo        pointer to an Inference Engine object
        @param address      shared memory address of the start of the data to read
        @param data         pointer to the buffer that will be filled with the returned data
        @param nelements    number of bytes to transfer
        @param card         FPGA card index
    */
    void IECOMPILER_API ie_read_data(void *cmemo, uint64_t address, void *data, uint64_t nelements, int card);

    /*!
    write data to an address in shared memory.
        @param cmemo        pointer to an Inference Engine object
        @param address      shared memory address of the location to write the data
        @param data         pointer to the data to write
        @param nelements    number of bytes to transfer
        @param card         FPGA card index
    */
    void IECOMPILER_API ie_write_data(void *cmemo, uint64_t address, const void *data, uint64_t nelements, int card);

    /*!
    write weights to an address in shared memory.
        @param cmemo        pointer to an Inference Engine object
        @param weight       array of weights
        @param bias         array of bias
        @param wsize        number of elements in 'weight' array
        @param bsize        number of elements in 'bias' array
        @param nid          id of the layer for which the weights are being overwritten. -1 is the last linear layer
     */
    void IECOMPILER_API ie_write_weights(void *cmemo, float *weight, float *bias, int wsize, int bsize, int nid);

    /*!
    create a MainMem for an FPGA card and initialize the FPGA (pico obj).
        @param cmemo        pointer to an Inference Engine object
        @param nfpga        number of FPGAs to use and initialize
        @param nclus        number of clusters to use
        @param fbitfile     pathname of the bitfile to load into the FPGA
    */
    void IECOMPILER_API ie_create_memcard(void *cmemo, int nfpga, int nclus, const char *fbitfile);

    /*!
    return an array with nonlinear coefficients (can be freed with free)
        @param cmemo        pointer to an Inference Engine object
        @param type         unused.   Type is always SFT_RELU.
    */
    IECOMPILER_API SF_INT *ie_get_nonlin_coefs(void *cmemo, int type);

    /*!
    create MemData, add to cmem, return its address: use address to read/write data to memory
        @param cmemo        pointer to an Inference Engine object
        @param len          number of words to allocate
        @param type         size of each word in bytes
        @param card         selects which FPGA card to use to allocate memory
        @param comment      comment for allocation, can be used in ASM code, prefixed with @
    */
    uint64_t IECOMPILER_API ie_malloc(void *cmemo, unsigned len, size_t type, int card, const char *comment);

    /*!
    read code from text file, generate assembly and return assembly
        @param cmemo        pointer to an Inference Engine object
        @param fname        text file path containing program
        @param instr_addr   memory address of instructions
        @param programlen   the generated program length in bytes is returned here
        @return  buffer with machine code instructions, to be freed with free
    */
    IECOMPILER_API uint32_t *ie_readcode(void *cmemo, const char *fname, uint64_t instr_addr, uint64_t *programlen);

    /*!
    set initial instructions, and start hw and poll/wait, return error or success
        @param cmemo        pointer to an Inference Engine object
        @param instr_addr   memory address of instructions
        @param hwtime       returns amount of time to run the accelerator
        @param mvdata       returns amount of data transferred to accelerator
        @param outsize      wait for this amount of data to return from accelerator. if 0 then wait for 2 sec
    */
    void IECOMPILER_API ie_hwrun(void *cmemo, uint64_t instr_addr, double *hwtime, double *mvdata, int outsize);

    /*!
    Loads multiple bitfiles without initializing hardware
        @param cmemo   pointer to an Inference Engine object
        @param inbins  array of pathnames to the bitfiles to load
        @param count   number of bitfiles to load
        @return pointer to an Inference Engine object to pass to ie_init
    */
    void IECOMPILER_API *ie_loadmulti(void *cmemo, const char *const *inbins, unsigned count);

    /*!
    Start training of a linear layer
    args:
        nin: number of input elements of the linear layer
        nout: number of output elements of the linear layer
        batch: number of input/output vectors to train in one shot
        A: starting weights matrix of nout x nin size
        b: starting bias vector of nout size
        Ashift: number of rational bits for A when converting to int
        Xshift: number of rational bits for input when converting to int
        Yshift: number of rational bits for output when converting to int
        Ygshift: number of ration bits for gradient when converting to int (used only in external gradient calculation)
        rate: learning rate; if 0, gradient will be calculated externally; if > 0, it will be the learning rate with LMS loss calculated internally
    */
    int IECOMPILER_API ie_trainlinear_start(void *cmemo, int nin, int nout, int batch, const float *A, const float *b, int Ashift, int Xshift, int Yshift, int Ygshift, float rate);
    /*!
    Pass training data
    args:
        X: input matrix of nin x batch size
        Y0: desired matrix of nout x batch size in internal gradient calculation;
            gradient of nout x batch size in external gradient calculation
        idx: arbitrary index where to store in memory

    Note:
    All training data can be stored in memory at different indexes only at the beginning, so it won't be required
    to store it at each iteration
    In internal gradient calculation mode, both X and Y0 can be stored at the beginning
    In external gradient calculation mode, only X can be stored (Y0 must be NULL) at the beginning as the gradient
    will have to be calculated at each iteration externally; in this case X will be NULL
    */
    int IECOMPILER_API ie_trainlinear_data(void *cmemo, const float *X, const float *Y0, int idx);
    /*!
    Run a training step in HW
    args:
        idx: index in memory where to get training data
    */
    int IECOMPILER_API ie_trainlinear_step(void *cmemo, int idx);
    /*!
    Run a training step in sw using SF_INTs

    Note:
    The results here should be numerically identical to HW mode; this routine is provided for correctness checking
    In software mode training data cannot be preloaded, so no idx is provided
    Only internal gradient calculation is supported here
    */
    int IECOMPILER_API ie_trainlinear_step_sw(void *cmemo);
    /*!
    Run a training step in sw using floats

    Note:
    In software mode training data cannot be preloaded, so no idx is provided
    Only internal gradient calculation is supported here
    */
    int IECOMPILER_API ie_trainlinear_step_float(void *cmemo);
    /*!
    Get the inference result for external gradient mode
    args:
        Y: Inference result of nout x batch size
    */
    int IECOMPILER_API ie_trainlinear_getY(void *cmemo, float *Y);
    /*!
    Get the learned matrices A and b
    args:
        A: learned weights matrix of nout x nin size
        b: learned bias vector of nout size
    */
    int IECOMPILER_API ie_trainlinear_get(void *cmemo, float *A, float *b);
    /*!
    Terminate the training process freeing all the resources used for training (ie_free will have to be called, too)
    */
    int IECOMPILER_API ie_trainlinear_end(void *cmemo);

    /*#ifdef __cplusplus
    }
    #endif
    */
}
static inline void *ie_safecreate()
{
    char version[10];
    void *cmemo = ie_create();

    if (ie_getinfo(cmemo, "version", version, 10) != 1)
    {
        fprintf(stderr, "Wrong libmicrondla.so version\n");
        exit(-1);
    }
    if (strcmp(version, microndla_version))
    {
        fprintf(stderr, "Wrong libmicrondla.so version, expecting %s, found %s\n", microndla_version, version);
        exit(-1);
    }
    return cmemo;
}

#endif

// Consolidation from additional .h files
class node_t;

namespace thnets
{

    /*  enum therror
      {
          ERR_OPENFILE = -1,
          ERR_READFILE = -2,
          ERR_NOTIMPLEMENTED = -3,
          ERR_CORRUPTED = -4,
          ERR_WRONGOBJECT = -5
      };
  */
    enum THDATATYPE
    {
        DT_UNDEFINED,
        DT_FLOAT,
        DT_UBYTE,
        DT_BYTE,
        DT_USHORT,
        DT_SHORT,
        DT_UINT,
        DT_INT,
        DT_ULONG,
        DT_LONG,
        DT_BOOL,
        DT_DOUBLE
    };

    typedef struct THNStorage
    {
        void *data; // size is datasize*size
        enum THDATATYPE datatype;
        char datasize;
        long size;
        int nref, mustfree; // mustfree = 0 (allocated somewhere else), 1 (free), 2 (cuda free)
    } THNStorage;

#define MAX_DIM 6 // Maximum number of supported dimensions

    typedef struct THNTensor
    {
        THNTensor();
        THNTensor(THDATATYPE datatype, const std::string &name = std::string());
        ~THNTensor();
        THNTensor(const THNTensor &t)
        {
            memcpy(size, t.size, sizeof(size));
            memcpy(stride, t.stride, sizeof(stride));
            nDimension = t.nDimension;
            storage = t.storage;
            if (storage)
                THAtomicIncrement(&storage->nref);
            storageOffset = t.storageOffset;
            bufferid = t.bufferid;
            datatype = t.datatype;
            datasize = t.datasize;
            name = t.name;
        }
        void print();
        // Placeholder for optional inputs.
        static THNTensor *Optional;

        long size[MAX_DIM];
        long stride[MAX_DIM];
        int nDimension;
        THNStorage *storage;
        long storageOffset;
        int bufferid;
        enum THDATATYPE datatype;
        char datasize;
        std::string name;
#ifdef LOWP
        float sub, mult;
#endif
    } THNTensor;

    enum AutoPadType
    {
        NOTSET = 0,
        VALID = NOTSET,
        SAME_UPPER = 1,
        SAME_LOWER = 2,
    };

    struct SpatialBase
    {
        THNTensor *bias, *weight;
        union
        {
            THNTensor *finput;    // SpatialConvolution
            THNTensor *addBuffer; // Linear
        };
    };

    struct SpatialConvolution : SpatialBase
    {
        int nOutputPlane, nInputPlane; // Channel
        int kZ, kH, kW;                // Kernel size
        int padZ, padH, padW;          // Padding at start
        int padZ2, padH2, padW2;       // Padding at end
        int dZ, dH, dW;                // Stride
        int dlZ, dlH, dlW;             // Dilations
        int refl_pad;
        AutoPadType autopad; // ONNX: 0 = VALID, 1 = SAME_UPPER, 2 = SAME_LOWER
        int groups;
        int activation;
    };

    struct SpatialFullConvolution : SpatialConvolution
    {
        int opadZ, opadH, opadW; // Output_padding
        struct
        {
            int output_rank;
            long output_shape[MAX_DIM];
        }; // Output_shape
        int adjZ, adjH, adjW;
        THNTensor *ones, *columns;
    };

    struct SpatialPooling
    {
        int kZ, kH, kW;          // Kernel size
        int padZ, padH, padW;    // Padding at start
        int padZ2, padH2, padW2; // Padding at end
        int dZ, dH, dW;          // Stride
        int autopad;             // ONNX: 0 = VALID, 1 = SAME_UPPER, 2 = SAME_LOWER
        int ceil_mode;
    };

    struct SpatialMaxPooling : SpatialPooling
    {
        int iwidth, iheight;
        THNTensor *indices;
    };

    struct SpatialAveragePooling : SpatialPooling
    {
        int count_include_pad;
    };

    struct Linear : SpatialBase
    {
        int commute; // Used for ONNX, if 1, invert A and B
        struct
        {
            float alpha, beta;
            int transA, transB;
        }; // Gemm
    };

    struct Threshold
    {
        float threshold, val, alpha;
        int inplace;
        struct
        {
            THNTensor *min, *max;
        }; // Clip
    };

    struct View
    {
        int numElements, nDimension;
        long size[MAX_DIM];
        struct
        {
            int axis;
        }; // Flatten
        struct
        {
            int axis_start, axis_count;
        }; // Reshape
    };

    struct Dropout
    {
        float p;
        int inplace, v2;
    };

    struct SpatialZeroPadding
    {
        int pad_l, pad_r, pad_t, pad_b;
    };

    struct Reshape
    {
        int numElements, batchMode;
        long size[MAX_DIM], batchsize[MAX_DIM];
        int nsize, nbatchsize;
    };

    struct SpatialBatchNormalization
    {
        THNTensor *running_mean, *running_var, *weight, *bias;
        double eps;
        bool inv;
        bool isplainadd; // This is an Add operation, weight is a vector of 1
        bool isplainmul; // This is a Mul operation
        bool isconst;    // This is a Const operation Tensor are const
    };

    struct Concat
    {
        struct network *net;
        union
        {
            int axis, dimension;
        };
    };

    struct Cast
    {
        THDATATYPE to;
    };

    struct ConstantOfShape
    {
        THNTensor *value;
    };

    struct Gather
    {
        int axis;
    };

    struct InstanceNormalization
    {
        float epsilon;
        THNTensor *scale, *bias;
    };

    struct LRN
    {
        float alpha, beta, bias;
        struct
        {
            size_t nsize;
            long size[MAX_DIM];
        };
    };

    struct Sequential
    {
        struct network *net;
    };

    struct PReLU
    {
        THNTensor *weight;
        int nOutputPlane;
    };

    struct Padding
    {
        float dim, pad, nInputDim, index, value;
    };

    struct Slice
    {
        struct
        {
            size_t naxes;
            long starts[MAX_DIM], ends[MAX_DIM], axes[MAX_DIM], steps[MAX_DIM];
        };
    };

    enum UpsampleAlgorithm
    {
        nearest,
        bilinear
    };

    struct Upsample
    {
        float width_scale, height_scale;
        int width, height;
        THNTensor *weight;
        UpsampleAlgorithm algorithm;
    };

    struct LSTM
    {
        THNTensor *W, *R, *B;
        int activations[3];
    };

    struct GRU
    {
        THNTensor *W, *R, *B;
        int activations[2];
    };

    struct Squeeze
    {
        struct
        {
            size_t naxes;
            int axes[MAX_DIM];
        };
    };

    struct Tile
    {
        struct
        {
            size_t nrepeats;
            long repeats[MAX_DIM];
        };
    };

    struct NonMaxSuppression
    {
        long center_point_box;
        // Optional tensors: these are scalars.
        long max_output_boxes_per_class;
        float iou_threshold, score_threshold;
        bool have_score_threshold;
    };

    struct Pad
    {
        int npads;
        long pads[MAX_DIM * 2];
    };

    struct Reduce
    {
        struct
        {
            size_t naxes;
            int axes[MAX_DIM];
        };
        int keepdims;
        int noop_with_empty_axes; // ONNX ReduceSum Opset-13 only
        int select_last_index;    // ArgMax
    };

    struct RoiAlign
    {
        enum RoiAlignMode
        {
            avg = 0,
            max,
        } mode;
        int output_height, output_width, sampling_ratio;
        float spatial_scale;
    };

    struct Scatter
    {
        int axis;
    };

    struct TopK
    {
        int axis, largest, sorted;
    };

    enum moduletype
    {
        MT_UNDEFINED,
        MT_SpatialConvolutionMM,
        MT_SpatialConvolutionVirtMM,
        MT_SpatialConvolution,
        MT_SpatialMaxPooling,
        MT_SpatialAveragePooling,
        MT_Linear,
        MT_SoftMax,
        MT_Threshold,
        MT_View,
        MT_Dropout,
        MT_SpatialZeroPadding,
        MT_Reshape,
        MT_Normalize,
        MT_SpatialFullConvolution,
        MT_SpatialMaxUnpooling,
        MT_SpatialBatchNormalization,
        MT_Sequential,
        MT_Concat,
        MT_ConcatTable,
        MT_JoinTable,
        MT_CAddTable,
        MT_CSubTable,
        MT_PReLU,
        MT_Identity,
        MT_Padding,
        MT_LogSoftMax,
        MT_Slice,
        MT_Cmax,
        MT_Upsample,
        MT_LSTM,
        MT_GRU,
        MT_Squeeze,
        MT_Unsqueeze,
        MT_Sigmoid,
        MT_SiLU,
        MT_Tanh,
        MT_Transpose,
        MT_DepthwiseConvolution,
        MT_DepthwiseTransposeConvolution,
        MT_CMulTable,
        MT_Elu,
        MT_Clip,
        MT_Tile,
        MT_Abs,
        MT_Add,
        MT_And,
        MT_Cast,
        MT_Ceil,
        MT_ConstantOfShape,
        MT_Copy,
        MT_Cos,
        MT_Div,
        MT_Equal,
        MT_Exp,
        MT_Erf,
        MT_Expand,
        MT_Flatten,
        MT_Floor,
        MT_Gather,
        MT_Greater,
        MT_GreaterEqual,
        MT_InstanceNormalization,
        MT_Log,
        MT_Log2,
        MT_LRN,
        MT_Less,
        MT_LessEqual,
        MT_Mul,
        MT_Max,
        MT_Min,
        MT_NonMaxSuppression,
        MT_NonZero,
        MT_NotEqual,
        MT_Neg,
        MT_Not,
        MT_Or,
        MT_Pad,
        MT_Pow,
        MT_Recip,
        MT_ReduceAll,
        MT_ReduceAny,
        MT_ReduceArgMax,
        MT_ReduceArgMin,
        MT_ReduceMax,
        MT_ReduceMean,
        MT_ReduceMin,
        MT_ReduceProd,
        MT_ReduceSum,
        MT_RoiAlign,
        MT_Round,
        MT_Rsqr,
        MT_Rsqrt,
        MT_ScatterElements,
        MT_Shape,
        MT_Sign,
        MT_Sin,
        MT_Sqr,
        MT_Sqrt,
        MT_Sub,
        MT_Sum,
        MT_Tan,
        MT_TopK,
        MT_Xor,
        MT_SoftPlus
    };
    struct network;

    struct module
    {
        module();
        ~module();

        // Get input IDX of this module.
        THNTensor *getInput(int idx);

        moduletype type;
        void (*inferShape)(module *mod);
        THNTensor *(*updateOutput)(struct module *m, THNTensor *in);
        void (*nnfree)(struct module *m);
        struct network *net;
        node_t *dla_node; // first DLA node_t that will run this operations
        int step;         // step in the CPU-DLA-CPU-DLA sequence, defines subgraphs
#ifdef OPENCL
        cl_kernel kernel;
        int clstatus;
#endif
        // These are currently used only by ONNX
        // They are always present in order not to require to define ONNX
        // when including this header
        int outidx[3];                       //! Output index+1 if > 0, otherwise this is not a network output (we have only LSTM with 3 outputs and GRU with 2 outputs)
        std::vector<int> inputs;             //! Indices of modules that outputs to the inputs to this module; the 8 MSBs contain the index of the output (e.g 3 outputs (LSTM), this output index can be 0, 1 or 2)
        std::vector<char *> inputnames;      //! Names of the inputs
        std::vector<THNTensor *> all_inputs; //! Shapes of all the inputs (also constant inputs)
        std::vector<THNTensor *> outputs;    //! Output of this module
        std::vector<int> consumers;          //! Indices of modules that consumes outputs of this module; the 8 MSBs contain the output index of this module
        // End ONNX
        union
        {
            struct SpatialConvolution SpatialConvolution;
            struct SpatialMaxPooling SpatialMaxPooling;
            struct SpatialAveragePooling SpatialAveragePooling;
            struct Linear Linear;
            struct Threshold Threshold;
            struct View View;
            struct Dropout Dropout;
            struct SpatialZeroPadding SpatialZeroPadding;
            struct Reshape Reshape;
            struct SpatialFullConvolution SpatialFullConvolution;
            struct SpatialBatchNormalization SpatialBatchNormalization;
            struct Sequential Sequential;
            struct Concat Concat;
            struct Sequential ConcatTable;
            struct Concat JoinTable;
            struct PReLU PReLU;
            struct Slice Slice;
            struct Upsample Upsample;
            struct LSTM LSTM;
            struct GRU GRU;
            struct Squeeze Squeeze;
            struct Tile Tile;
            struct Cast Cast;
            struct ConstantOfShape ConstantOfShape;
            struct Gather Gather;
            struct InstanceNormalization InstanceNormalization;
            struct LRN LRN;
            struct NonMaxSuppression NonMaxSuppression;
            struct Pad Pad;
            struct Reduce Reduce;
            struct RoiAlign RoiAlign;
            struct Scatter Scatter;
            struct TopK TopK;
        };
    };

    int getoutput(struct network *net, const std::string &name);
    int getoutput_c(struct network *net, const char *name);

    enum th_engine
    {
        ENGINE_CPU,
        ENGINE_CUDA,
        ENGINE_OPENCL,
        ENGINE_OPENCLINIT,
        ENGINE_LOWP,
        ENGINE_TABLE
    };

    struct PINDESC
    {
        PINDESC() : thidx(-1), dlaidx(-1) {}
        PINDESC(int thidx) : thidx(thidx), dlaidx(-1) {}
        int thidx;        //! thnets index (if positive: 8 MSBs is output index, 24 LSBs is module index; if negative: input index)
        int dlaidx;       //! DLA buffer index
        std::string name; //! output name
    };

    //! Each subgraph identifies the modules that run sequentially on the same device (CPU of DLA)
    struct subgraph
    {
        subgraph() : dla_node(0), step(-1) {}
        subgraph(int step) : dla_node(0), step(step) {}
        node_t *dla_node;             //! If not NULL, the first node_t of the subgraph (for DLA subgraphs), otherwise it's a CPU subgraph
        int step;                     //! corresponds to step in node_t
        std::vector<int> modules;     //! Indices to CPU modules that make up this subgraph
        std::vector<PINDESC> inputs;  //! Inputs to this subgraph
        std::vector<PINDESC> outputs; //! Outputs of this subgraph

        bool isinput(int thidx)
        {
            for (PINDESC &pin : inputs)
                if (pin.thidx == thidx)
                    return true;
            return false;
        }
    };

    struct bistring
    {
        std::string from, to;
    };

    struct network
    {
        network(th_engine engine, int nalloc);
        ~network();

        th_engine engine;
        int nelem, nalloc;
        int steps; // Number of subgraphs (steps in DLA-CPU switching)
        module *modules;
        std::string inshapes;                  // String describing input shapes
        std::vector<thnets::THNTensor *> in_t; // tensors describing inputs shapes

        std::string onnx_domain; // (ONNX only) opset domain
        long onnx_version;       // (ONNX only) opset version

        std::map<std::string, THNTensor> tensors; // list of tensors in the network
        std::vector<std::string> inputs;          // list of input tensors
        std::vector<std::string> outputs;         // list of output tensors
        std::vector<subgraph *> subgraphs;        // list of subgraphs in the network
        std::vector<PINDESC> dlainputs;           // merged subgraphs inputs
        std::list<bistring> inputs_aliases;       // alternative input names map to real inputs (preprocessing layers are not saved as layers, so their outputs become thnets network input)
        float mul[3], add[3];                     // Input normalization for images

        // Get network input index. Inputs are encoded as negative numbers,
        // the first input is -1, the second -2, ...
        // Return 0 if name is not an input.
        int getInput(const std::string &name) const
        {
            for (size_t i = 0; i < inputs.size(); ++i)
                if (inputs[i] == name)
                    return -1 - i;
            for (bistring bs : inputs_aliases)
                if (bs.from == name)
                {
                    for (size_t i = 0; i < inputs.size(); ++i)
                        if (inputs[i] == bs.to)
                            return -1 - i;
                }
            return 0;
        }

        // Get tensor by name. Returns NULL if not found.
        THNTensor *getTensor(const std::string &name)
        {
            auto iter = tensors.find(name);
            if (iter == tensors.end())
                return 0;
            return &iter->second;
        }
        const THNTensor *getTensor(const std::string &name) const
        {
            auto iter = tensors.find(name);
            if (iter == tensors.end())
                return 0;
            return &iter->second;
        }

        bool isInitializer(const std::string &name) const
        {
            if (const THNTensor *tensor = getTensor(name))
                return tensor->storage != 0;
            return false;
        }
        THNTensor *getInitializer(const std::string &name)
        {
            if (THNTensor *tensor = getTensor(name))
                return tensor->storage != 0 ? tensor : 0;
            return 0;
        }
    };

    // extern jmp_buf therror_env;
    extern char therror[1000];

    double TableGetNumber(struct table *t, const char *name);
    int TableGetBoolean(struct table *t, const char *name);
    THNTensor *TableGetTensor(struct table *t, const char *name);
    void *TableGetStorage(struct table *t, const char *name, int *nelem);
    [[noreturn]] void THError(const char *fmt, ...);
    THNTensor *THNTensor_new(enum THDATATYPE datatype);
    THNTensor *THNTensor_new(enum THDATATYPE datatype, const std::string &name);
    THNTensor *THNTensor_new(THDATATYPE datatype, const char *name);

    // api for create thnets externally (abi=0 compatible)
    network *create_network(int g_size);
    void thnetwork_add_input(struct network *net, const char *name);
    void thnetwork_add_output(struct network *net, const char *name);

    THNStorage *THNStorage_new(long size, enum THDATATYPE datatype);
    THNStorage *THNStorage_newwithbuffer(void *buffer, enum THDATATYPE datatype, long buffersize);
    THNTensor *THNTensor_newWithStorage1d(THNStorage *storage, long storageOffset, long size0, long stride0);
    THNTensor *THNTensor_newWithStorage2d(THNStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1);
    THNTensor *THNTensor_newWithStorage3d(THNStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1, long size2, long stride2);
    THNTensor *THNTensor_newWithStorage4d(THNStorage *storage, long storageOffset, long size0, long stride0, long size1, long stride1, long size2, long stride2,
                                          long size3, long stride3);
    THNTensor *THNTensor_newWithTensor(THNTensor *tensor);
    bool THNTensor_isZero(THNTensor *t);
    void THNTensor_defaultStrides(THNTensor *t);
    void THNTensor_transpose(THNTensor *tdst, THNTensor *tsrc, int dimension1, int dimension2);
    THNTensor *THNTensor_newTranspose(THNTensor *tensor, int dimension1_, int dimension2_);
    void *THNTensor_data(THNTensor *tensor);
    float *THNTensor_fdata(THNTensor *tensor);
    int THNTensor_isSameSizeAs(const THNTensor *self, const THNTensor *src);
    const std::string THNTensor_Shape(THNTensor *t);
    void THNTensor_allocate(THNTensor *t);
    void THNTensor_resize(THNTensor *t, long *size, int nDimension);
    void THNTensor_resizeNoStorage(THNTensor *t, long *size, int nDimension);
    void THNTensor_resize4d(THNTensor *t, long size0, long size1, long size2, long size3);
    void THNTensor_resize3d(THNTensor *t, long size0, long size1, long size2);
    void THNTensor_resize2d(THNTensor *t, long size0, long size1);
    void THNTensor_resize1d(THNTensor *t, long size0);
    void THNTensor_resizeAs(THNTensor *tdst, THNTensor *tsrc);
    long THNTensor_nElement(THNTensor *t);
    void THNTensor_set(THNTensor *tdst, THNTensor *tsrc);
    void THNTensor_zero(THNTensor *t);
    void THNTensor_fill(THNTensor *t, float value);
    void THNTensor_copy(THNTensor *tdst, THNTensor *tsrc);
    void THNTensor_safecopy(THNTensor *tdst, THNTensor *tsrc);
    void THNTensor_slice(THNTensor *dst, THNTensor *src, int dimension, long from, long to);
    void THNTensor_free(THNTensor *t);
    void THNStorage_free(THNStorage *s);
    THNTensor *THNTensor_newSelect(THNTensor *tensor, int dimension, long sliceIndex);
    THNTensor *THNTensor_squeeze(THNTensor *t);
    double THExpMinusApprox(double x);
    void THBlas_gemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc);
    void THNTensor_addmm(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *m1, THNTensor *m2);
    void THNTensor_convmm(THNTensor *r, float beta, float alpha, THNTensor *filt, THNTensor *m,
                          int kH, int kW, int dH, int dW, int padH, int padW);
    void THNTensor_addr(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *vec1, THNTensor *vec2);
    void THNTensor_addmv(THNTensor *r_, float beta, THNTensor *t, float alpha, THNTensor *mat, THNTensor *vec);
    void THNTensor_conv2Dmm(THNTensor *r_, float beta, float alpha, THNTensor *t_, THNTensor *k_, long srow, long scol, const char *vf, const char *xc);
    void THNTensor_conv2Dmv(THNTensor *r_, float beta, float alpha, THNTensor *t_, THNTensor *k_, long srow, long scol, const char *vf, const char *xc);

#define thfmaxf(a, b) ((a) > (b) ? (a) : (b))
#define thfminf(a, b) ((a) < (b) ? (a) : (b))
#define TH_MODULEIDX(a) ((a) >= 0 ? (a)&0xffffff : (a)) // modules[].inputs[] contains the moduleidx & module output in the 8 MSBs
#define TH_OUTIDX(a) ((a) >= 0 ? (a) >> 24 : (a))
#define THIDX_NONE (-10000)
#define TH_INPUTMODULE(m) (((m)->inputs[0]) >= 0 ? m->net->modules + TH_MODULEIDX((m)->inputs[0]) : 0) // First input module of m or 0

#define THInf FLT_MAX

#ifdef HAVEFP16
    void tofp16(__fp16 *dst, const float *src, size_t len);
    void fromfp16(float *dst, const __fp16 *src, size_t len);
#endif

    /* High level API */

    typedef struct network THNETWORK;

    void THInit();
    THNETWORK *THLoadNetwork(const char *path, const std::vector<THNTensor> &inputShapes);
    THNETWORK *THSimplify(THNETWORK *network);
    THNTensor *THForward(THNETWORK *net, THNTensor *in);
    void THMakeSpatial(THNETWORK *network, int size);
    int THProcessFloat(THNETWORK *network, float *data, int batchsize, int width, int height, int nplanes, float **result, int *outwidth, int *outheight);
    int THProcessImages(THNETWORK *network, unsigned char **images, int batchsize, int width, int height, int stride, float **result, int *outwidth, int *outheight, int bgr);
    int THProcessYUYV(THNETWORK *network, unsigned char *image, int width, int height, float **results, int *outwidth, int *outheight);
    THNETWORK *THCreateCudaNetwork(THNETWORK *net);
    THNETWORK *THCreateOpenCLNetwork(THNETWORK *net);
    THNETWORK *THCreateLowpNetwork(THNETWORK *net, float range);
    int THCudaHalfFloat(int enable);
    int THOpenCLHalfFloat(int enable);
    int THUseSpatialConvolutionMM(THNETWORK *network, int mm_type);
    void THFreeNetwork(THNETWORK *network);
    int THLastError();

#ifdef CUDNN
#include "cudnn/cudnn_th.h"
#endif

#ifdef OPENCL
#include "opencl/opencl_th.h"
#endif

#ifdef LOWP
#include "lowp/lowp.h"
#endif

#ifdef USEQSML
    void init_thnets4qsml_conv(THNETWORK *network);
    void transform_mem(struct module newmod, int col, int row, int plane, int outp);
    float *transform_mem_input(float *in1, int col, int row, int plane);
#endif
}
void thload_Conv2d(struct thnets::module *m, float *weight, float *bias,
                   int inp, int outp, int kW, int kH, int pW, int pH, int dW, int dH, int dlW, int dlH, int group);
void thload_TransposedConv2d(struct thnets::module *m, float *weight, float *bias,
                             int inp, int outp, int kW, int kH, int pW, int pH, int dW, int dH, int opW, int opH, int group);
void thload_Threshold(struct thnets::module *m);
void thload_Maxpool2d(struct thnets::module *m,
                      int kW, int kH, int pW, int pH, int dW, int dH, int dlW, int dlH, bool ceil);
void thload_Avgpool2d(struct thnets::module *m,
                      int kW, int kH, int pW, int pH, int dW, int dH, bool ceil);
void thload_Linear(struct thnets::module *m, float *weight, float *bias, int i, int o);
void thload_View(struct thnets::module *m);
void thload_Sigmoid(struct thnets::module *m);
void thload_SiLU(struct thnets::module *m);
void thload_Tanh(struct thnets::module *m);
void thload_BatchNorm(struct thnets::module *m, float *weight, float *bias, float *run_mean, float *run_var, float eps, int len);
void thload_Add(struct thnets::module *m);
void thload_Sub(struct thnets::module *m);
void thload_Concat(struct thnets::module *m, int dim);
void thload_Upsample(struct thnets::module *m, int w_scale, int h_scale);

#endif