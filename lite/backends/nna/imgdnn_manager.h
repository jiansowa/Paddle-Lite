// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "imgdnn.h"  // NOLINT
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace nna {

static inline
void CheckAndPrint(bool cond,
                   const char *msg,
                   int line,
                   const char *filename
                  ) {
  if (cond) {
    std::stringstream err_msg;
    err_msg << "ERROR: " << msg << "\n";
    err_msg << "Violated condition at line " << line << " in " << filename;
    std::cerr << err_msg.str() << "\n";
    exit(EXIT_FAILURE);
  }
}

#define ASSERT(statement, msg)  \
  lite::nna::CheckAndPrint(statement, msg, __LINE__, __FILE__)

class ImgdnnManager {
  imgdnn_err_code err_;
  imgdnn_device device_;
  imgdnn_network net_{nullptr};
  imgdnn_context context_{nullptr};
  imgdnn_binding binding_{nullptr};
  imgdnn_network_object net_obj_{nullptr};

  std::vector<uint8_t*> coef_pool;

  uint8_t * GetBufromPool(size_t size) {
    uint8_t* buf = new uint8_t[size];
    coef_pool.push_back(buf);
    return buf;
  }

 public:
  ImgdnnManager();

  virtual ~ImgdnnManager() {
    std::cout << "~ImgdnnManager called" << std::endl;
    if (net_obj_) err_ = imgdnnNetworkObjectDestroy(net_obj_);
    if (context_) err_ = imgdnnContextDestroy(context_);
    if (binding_) err_ = imgdnnBindingDestroy(binding_);
    if (net_) err_ = imgdnnNetworkDestroy(net_);

    for (auto buf : coef_pool) delete[] buf;
  }

  imgdnn_network GetNetwork() { return net_; }

  imgdnn_tensor createInputTensor(imgdnn_tensor_descriptor *desc) {
    return imgdnnNetworkInput(net_, desc, &err_);
  }

  imgdnn_tensor createFixedInputTensor(imgdnn_tensor_descriptor *desc,
                                       const void *const fixed_data,
                                       bool mem_copy) {
    imgdnn_tensor fixed_input;
    if (mem_copy) {
      size_t buffer_size = imgdnnGetDescriptorSize(desc, &err_);
      void *buf = GetBufromPool(buffer_size);
      memcpy(buf, fixed_data, buffer_size);
      fixed_input = imgdnnNetworkFixedInput(net_, desc, buf, &err_);
    } else {
      fixed_input = imgdnnNetworkFixedInput(net_, desc, fixed_data, &err_);
    }
    return fixed_input;
  }

  imgdnn_tensor createConvolutionLayer(imgdnn_tensor input_tensor,
                                       imgdnn_tensor weights_tensor,
                                       imgdnn_tensor bias_tensor,
                                       imgdnn_quant_param dst_quant_param,
                                       unsigned int stride[2],
                                       unsigned int pad_begin[2],
                                       unsigned int pad_end[2],
                                       unsigned int dilation[2],
                                       bool use_dwconv = false);
  imgdnn_tensor createBatchNormLayer(imgdnn_tensor input_tensor,
                                     const void *const avg_in,
                                     const void *const var_in,
                                     const float eps);
  imgdnn_tensor createPoolingLayer(imgdnn_tensor in_tensor,
                                   imgdnn_quant_param dst_quant_param,
                                   const unsigned int size[2],
                                   const unsigned int stride[2],
                                   const unsigned int pad_to_begin[2],
                                   const unsigned int pad_to_end[2],
                                   imgdnn_pooling_type type);
  imgdnn_tensor createFullyConnectedLayer(imgdnn_tensor input_tensor,
                                          imgdnn_tensor weights_tensor,
                                          imgdnn_tensor bias_tensor,
                                          imgdnn_quant_param dst_quant_param);
  imgdnn_tensor createSoftmaxLayer(imgdnn_tensor in_tensor,
                                   float beta,
                                   unsigned int axis,
                                   imgdnn_quant_param dst_quant_param);
  imgdnn_tensor createScaleLayer(imgdnn_tensor input_tensor,
                                 bool with_biasscale,
                                 const void *const scale,
                                 const void *const bias);

  imgdnn_tensor createReLULayer(imgdnn_tensor in_tensor,
                                  bool has_min_clamp,
                                  float min_clamp,
                                  bool has_max_clamp,
                                  float max_clamp,
                                  float negative_slope) {
    return imgdnnNetworkReLUOp(net_, in_tensor, has_min_clamp, min_clamp,
        has_max_clamp, max_clamp, negative_slope, &err_);
  }

  imgdnn_network_object createNetworkObject(unsigned int num_inputs,
                                            imgdnn_tensor *inputs,
                                            unsigned int num_outputs,
                                            imgdnn_tensor *outputs);

  imgdnn_memory importMemory(void* memory, size_t size,
      imgdnn_import_mem_type import_mem_type = IMGDNN_IMPORT_MEM_TYPE_CPU)  {
    imgdnn_memory mem = imgdnnImportMemory(context_, memory, size,
        import_mem_type, &err_);
    ASSERT(err_ != IMGDNN_SUCCESS, "ImportMemory fails");
    return mem;
  }

  imgdnn_memory allocateMemory(size_t size) {
    imgdnn_memory mem = imgdnnAllocateMemory(context_, size, &err_);
    ASSERT(err_ != IMGDNN_SUCCESS, "AllocateMemory fails");
    return mem;
  }

  void destroyMemory(imgdnn_memory memory)  {
    err_ = imgdnnMemoryDestroy(memory);
    ASSERT(err_ != IMGDNN_SUCCESS, "MemoryDestroy fails");
  }

  void* lockMemory(imgdnn_memory memory, imgdnn_lock_access lock_access)  {
    void *mem = imgdnnMemoryLock(memory, lock_access, &err_);
    ASSERT(err_ != IMGDNN_SUCCESS, "MemoryLock fails");
    return mem;
  }

  void unlockMemory(imgdnn_memory memory) {
    err_ = imgdnnMemoryUnlock(memory);
    ASSERT(err_ != IMGDNN_SUCCESS, "MemoryUnLock fails");
  }

  void getNetworkObjectInputs(unsigned int max_inputs, imgdnn_input inputs[],
      unsigned int *num_inputs) {
    ASSERT(net_obj_ == nullptr, "NetworkObject NULL when get its inputs");
    err_ = imgdnnNetworkObjectGetInputs(net_obj_, max_inputs, inputs,
        num_inputs);
    ASSERT(err_ != IMGDNN_SUCCESS, "NetworkObjectGetInputs failed!");
  }

  void getNetworkObjectOutputs(unsigned int max_outputs,
      imgdnn_output outputs[], unsigned int *num_outputs) {
    ASSERT(net_obj_ == nullptr, "NetworkObject NULL when get its outputs");
    err_ = imgdnnNetworkObjectGetOutputs(net_obj_, max_outputs, outputs,
        num_outputs);
    ASSERT(err_ != IMGDNN_SUCCESS, "NetworkObjectGetOutputs failed!");
  }

  imgdnn_tensor_descriptor getInputDescriptor(imgdnn_input input) {
    imgdnn_tensor_descriptor desc = imgdnnGetInputDescriptor(input, &err_);
    ASSERT(err_ != IMGDNN_SUCCESS, "GetInputDescriptors failed!");
    return desc;
  }

  imgdnn_tensor_descriptor getOutputDescriptor(imgdnn_output output) {
    imgdnn_tensor_descriptor desc = imgdnnGetOutputDescriptor(output, &err_);
    ASSERT(err_ != IMGDNN_SUCCESS, "GetOutputDescriptors failed!");
    return desc;
  }

  size_t getDescriptorSize(const imgdnn_tensor_descriptor *const descriptor)  {
    size_t size = imgdnnGetDescriptorSize(descriptor, &err_);
    ASSERT(err_ != IMGDNN_SUCCESS, "GetDescriptorSize failed!");
    return size;
  }

  void addBindingInput(imgdnn_input input, imgdnn_memory memory)  {
    err_ = imgdnnBindingAddInput(binding_, input, memory);
    ASSERT(err_ != IMGDNN_SUCCESS, "BindingAddInput failed!");
  }

  void addBindingOutput(imgdnn_output output, imgdnn_memory memory)  {
    err_ = imgdnnBindingAddOutput(binding_, output, memory);
    ASSERT(err_ != IMGDNN_SUCCESS, "BindingAddOutput failed!");
  }

  void executeNetworkObject(bool blocking_execute,
                           unsigned int num_events_in_wait_list,
                           const imgdnn_event event_wait_list[],
                           imgdnn_event *event) {
    err_ = imgdnnNetworkObjectExecute(net_obj_, binding_, blocking_execute,
        num_events_in_wait_list, event_wait_list, event);
    ASSERT(err_ != IMGDNN_SUCCESS, "NetworkObjectExecute failed!");
  }
};

}  // namespace nna
}  // namespace lite
}  // namespace paddle
