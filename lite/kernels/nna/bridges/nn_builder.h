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
namespace subgraph {
namespace nna {

class ConvNetBuilder {
  imgdnn_network net;
  imgdnn_err_code err;
  // imgdnn_network_object net_obj;
  // imgdnn_device device;
  // imgdnn_context context;

  std::vector<float *> wg_data;
  std::vector<float *> ba_data;
  std::vector<float *> avg_data;
  std::vector<float *> var_data;
  std::vector<float *> sc_data;
  std::vector<float *> bsc_data;
  // std::vector<float *> ew_data;

 public:
  ConvNetBuilder();

  virtual ~ConvNetBuilder() {
    imgdnnNetworkDestroy(net);
    // imgdnnNetworkObjectDestroy(net_obj);
    // imgdnnContextDestroy(context);

    // if (netblob)
    while (wg_data.size()) {
      free(wg_data.back());
      wg_data.pop_back();
    }
    while (ba_data.size()) {
      free(ba_data.back());
      ba_data.pop_back();
    }
    while (avg_data.size()) {
      free(avg_data.back());
      avg_data.pop_back();
    }
    while (var_data.size()) {
      free(var_data.back());
      var_data.pop_back();
    }
    while (sc_data.size()) {
      free(sc_data.back());
      sc_data.pop_back();
    }
    while (bsc_data.size()) {
      free(bsc_data.back());
      bsc_data.pop_back();
    }
  }

  imgdnn_network GetNetwork() { return net; }
  imgdnn_tensor createInputTensor(imgdnn_tensor_descriptor *desc) {
    return imgdnnNetworkInput(net, desc, &err);
  }
  imgdnn_tensor createFixedInputTensor(imgdnn_tensor_descriptor *desc,
                                       const void *const fixed_data) {
    return imgdnnNetworkFixedInput(net, desc, fixed_data, &err);
  }

  imgdnn_tensor createFixedInputTensorFloat(const void *const fixed_data,
                                            const lite::DDim &dims);
  imgdnn_tensor createFixedInputTensorQuantized(PrecisionType precision,
                                                const void *const fixed_data,
                                                const lite::DDim &dims,
                                                std::vector<float> scales,
                                                unsigned axis,
                                                unsigned channels);
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
  imgdnn_network_object createNetworkObject(imgdnn_device device,
                                            imgdnn_context context,
                                            unsigned int num_inputs,
                                            imgdnn_tensor *inputs,
                                            unsigned int num_outputs,
                                            imgdnn_tensor *outputs);
};

}  // namespace nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
