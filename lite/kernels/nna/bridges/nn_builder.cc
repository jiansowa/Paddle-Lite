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

#include "nn_builder.h"  // NOLINT
#include <utility>
#include "lite/kernels/nna/bridges/graph.h"
#include "lite/kernels/nna/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nna {

static void err_callback(imgdnn_report_flags flags,
                         const char **tensor_names,
                         int num_tensor_names,
                         imgdnn_err_code error_code,
                         const char *error_message) {
  std::string msg_prefix;
  switch (flags) {
    case imgdnn_report_flags::IMGDNN_REPORT_ERROR:
      msg_prefix = "ERROR";
      break;
    case imgdnn_report_flags::IMGDNN_REPORT_VERBOSE:
      msg_prefix = "VERBOSE";
      break;
    case imgdnn_report_flags::IMGDNN_REPORT_INFO:
      msg_prefix = "INFO";
      break;
    case imgdnn_report_flags::IMGDNN_REPORT_WARNING:
      msg_prefix = "WARNING";
      break;
    default:
      std::cerr << "unknown report flag in error callback" << std::endl;
  }

  std::cerr << msg_prefix << ": " << error_message << std::endl;
}

ConvNetBuilder::ConvNetBuilder() {
  // imgdnn_err_code errcode_ret;
  net = imgdnnCreateNetwork(&err);
  // CHECK(err == IMGDNN_SUCCESS)
  //  << "Create ImgdnnNetwork fails1";
  err = imgdnnSetErrorHandler(err_callback);
  // CHECK(err == IMGDNN_SUCCESS)
  //  << "Could not set callback for error handling";
}

imgdnn_tensor ConvNetBuilder::createFixedInputTensorFloat(
    const void *const fixed_data, const lite::DDim &dims) {
  imgdnn_tensor_descriptor desc;
  desc.type = IMGDNN_TYPE_F32;
  desc.dimensions = (unsigned)dims.size();
  for (uint32_t i = 0; i < dims.size(); ++i) desc.size[i] = dims[i];

  imgdnn_tensor out_tensor =
      imgdnnNetworkFixedInput(net, &desc, fixed_data, &err);
  return out_tensor;
}

imgdnn_tensor ConvNetBuilder::createFixedInputTensorQuantized(
    PrecisionType precision,
    const void *const fixed_data,
    const lite::DDim &dims,
    std::vector<float> scales,
    unsigned axis,
    unsigned channels) {
  imgdnn_tensor_descriptor desc;
  imgdnn_per_axis_quant_param *axis_qp;
  std::vector<int> zero_ps;
  if (precision == paddle::lite_api::PrecisionType::kInt8) {
    desc.type = IMGDNN_TYPE_QPA_I8;
    zero_ps.resize(scales.size());  // default 0
    axis_qp = imgdnnCreatePerAxisQuantParam(
        axis, channels, scales.data(), zero_ps.data());
    CHECK(axis_qp != nullptr);
    desc.quant_param.per_axis = axis_qp;
  }

  desc.dimensions = (unsigned)dims.size();
  for (uint32_t i = 0; i < dims.size(); ++i) desc.size[i] = dims[i];

  imgdnn_tensor out_tensor =
      imgdnnNetworkFixedInput(net, &desc, fixed_data, &err);
  imgdnnDestroyPerAxisQuantParam(desc.quant_param.per_axis);

  return out_tensor;
}

imgdnn_tensor ConvNetBuilder::createConvolutionLayer(
    imgdnn_tensor input_tensor,
    imgdnn_tensor weights_tensor,
    imgdnn_tensor bias_tensor,
    imgdnn_quant_param dst_quant_param,
    unsigned int stride[2],
    unsigned int pad_begin[2],
    unsigned int pad_end[2],
    unsigned int dilation[2],
    bool use_dwconv) {
  imgdnn_tensor convw_tensor;
  if (use_dwconv) {
    // transpose weight
    int order[4] = {1, 0, 2, 3};
    imgdnn_tensor transport_weights =
        imgdnnNetworkTransposeOp(net, weights_tensor, order, &err);
    convw_tensor = imgdnnNetworkDepthConvolution2dOp_v2(net,
                                                        input_tensor,
                                                        transport_weights,
                                                        stride,
                                                        pad_begin,
                                                        pad_end,
                                                        dilation,
                                                        &err);
  } else {
    convw_tensor = imgdnnNetworkConvolution2dOp_v2(net,
                                                   input_tensor,
                                                   weights_tensor,
                                                   stride,
                                                   pad_begin,
                                                   pad_end,
                                                   dilation,
                                                   &err);
  }

  // debug
  imgdnn_tensor_descriptor desc_1;
  imgdnnGetTensorDescriptor(input_tensor, &desc_1);
  imgdnnGetTensorDescriptor(weights_tensor, &desc_1);
  imgdnnGetTensorDescriptor(convw_tensor, &desc_1);

  imgdnn_tensor conv2d_tensor;
  if (bias_tensor) {
    imgdnn_tensor convw_int_tensor =
        imgdnnNetworkCastOp(net, convw_tensor, IMGDNN_TYPE_I32, nullptr, &err);

    imgdnn_tensor_descriptor bias_desc;
    imgdnnGetTensorDescriptor(convw_tensor, &bias_desc);

    imgdnn_tensor broadcast2_tensor;
    broadcast2_tensor =
        imgdnnNetworkBroadcastOp(net, bias_tensor, 2, bias_desc.size[2], &err);

    imgdnn_tensor broadcast3_tensor;
    broadcast3_tensor = imgdnnNetworkBroadcastOp(
        net, broadcast2_tensor, 3, bias_desc.size[3], &err);

    conv2d_tensor = imgdnnNetworkBinaryOp(
        net, convw_int_tensor, broadcast3_tensor, IMGDNN_OPERATION_ADD, &err);
  } else {
    conv2d_tensor = convw_tensor;
  }

  imgdnn_tensor conv2d_out_tensor;
  imgdnn_tensor_descriptor desc;
  imgdnnGetTensorDescriptor(input_tensor, &desc);
  if (desc.type == IMGDNN_TYPE_Q_I8 || desc.type == IMGDNN_TYPE_Q_U8) {
    conv2d_out_tensor = imgdnnNetworkCastOp(
        net, conv2d_tensor, desc.type, &dst_quant_param, &err);
  }

  return conv2d_out_tensor;
}

imgdnn_tensor ConvNetBuilder::createBatchNormLayer(imgdnn_tensor input_tensor,
                                                   const void *const avg_in,
                                                   const void *const var_in,
                                                   const float eps) {
  imgdnn_tensor bna_tensor;
  imgdnn_tensor average_tensor;
  imgdnn_tensor_descriptor av_desc;

  imgdnn_tensor broadcast2_tensor;
  imgdnn_tensor broadcast3_tensor;

  unsigned int buffer_size;

  imgdnn_tensor_descriptor in_desc;
  imgdnnGetTensorDescriptor(input_tensor, &in_desc);

  av_desc.dimensions = 2;
  av_desc.type = in_desc.type;
  av_desc.size[0] = in_desc.size[0];
  av_desc.size[1] = in_desc.size[1];

  buffer_size = imgdnnGetDescriptorSize(&av_desc, &err);
  avg_data.push_back(reinterpret_cast<float *>(calloc(1, buffer_size)));
  memcpy(avg_data.back(), avg_in, buffer_size);
  average_tensor =
      imgdnnNetworkFixedInput(net, &av_desc, avg_data.back(), &err);

  broadcast2_tensor =
      imgdnnNetworkBroadcastOp(net, average_tensor, 2, in_desc.size[2], &err);

  broadcast3_tensor = imgdnnNetworkBroadcastOp(
      net, broadcast2_tensor, 3, in_desc.size[3], &err);

  bna_tensor = imgdnnNetworkBinaryOp(
      net, input_tensor, broadcast3_tensor, IMGDNN_OPERATION_SUB, &err);

  imgdnn_tensor variance_tensor;
  imgdnn_tensor_descriptor va_desc;

  va_desc.dimensions = 2;
  va_desc.type = in_desc.type;
  va_desc.size[0] = in_desc.size[0];
  va_desc.size[1] = in_desc.size[1];

  buffer_size = imgdnnGetDescriptorSize(&va_desc, &err);
  var_data.push_back(reinterpret_cast<float *>(calloc(1, buffer_size)));
  memcpy(var_data.back(), var_in, buffer_size);

  // Perform 1/sqrt(var+eps) and Update var_data.
  {
    buffer_size /= sizeof(float);
    float *variance = var_data.back();
    for (size_t i = 0; i < buffer_size; i++) {
      variance[i] = 1.0 / (sqrt(variance[i] + eps));
    }
  }

  variance_tensor =
      imgdnnNetworkFixedInput(net, &va_desc, var_data.back(), &err);

  broadcast2_tensor =
      imgdnnNetworkBroadcastOp(net, variance_tensor, 2, in_desc.size[2], &err);

  broadcast3_tensor = imgdnnNetworkBroadcastOp(
      net, broadcast2_tensor, 3, in_desc.size[3], &err);

  imgdnn_tensor bn_tensor;
  bn_tensor = imgdnnNetworkBinaryOp(
      net, bna_tensor, broadcast3_tensor, IMGDNN_OPERATION_MUL, &err);

  return bn_tensor;
}

imgdnn_tensor ConvNetBuilder::createPoolingLayer(
    imgdnn_tensor in_tensor,
    imgdnn_quant_param dst_quant_param,
    const unsigned int size[2],
    const unsigned int stride[2],
    const unsigned int pad_to_begin[2],
    const unsigned int pad_to_end[2],
    imgdnn_pooling_type type) {
  // debug
  imgdnn_tensor_descriptor desc_1;
  imgdnnGetTensorDescriptor(in_tensor, &desc_1);

  imgdnn_tensor pool_tensor = imgdnnNetworkPooling2dOp_v2(
      net, in_tensor, size, stride, pad_to_begin, pad_to_end, type, &err);
  // debug
  imgdnnGetTensorDescriptor(pool_tensor, &desc_1);

  imgdnn_tensor_descriptor desc;
  imgdnnGetTensorDescriptor(in_tensor, &desc);
  if (desc.type == IMGDNN_TYPE_Q_I8 || desc.type == IMGDNN_TYPE_Q_U8) {
    pool_tensor = imgdnnNetworkCastOp(
        net, pool_tensor, desc.type, &dst_quant_param, &err);
  }

  return pool_tensor;
}

imgdnn_tensor ConvNetBuilder::createFullyConnectedLayer(
    imgdnn_tensor input_tensor,
    imgdnn_tensor weights_tensor,
    imgdnn_tensor bias_tensor,
    imgdnn_quant_param dst_quant_param) {
  imgdnn_tensor fcw_tensor;
  imgdnn_tensor fcb_tensor;

  imgdnn_tensor_descriptor in_desc;
  imgdnnGetTensorDescriptor(input_tensor, &in_desc);

  for (unsigned i = 2; i < in_desc.dimensions; ++i)
    in_desc.size[1] *= in_desc.size[i];
  in_desc.dimensions = 2;

  auto reshaped_input =
      imgdnnNetworkReshapeOp(net, input_tensor, &in_desc, &err);

  // debug
  imgdnn_tensor_descriptor desc_1;
  imgdnnGetTensorDescriptor(reshaped_input, &desc_1);
  imgdnn_tensor_descriptor desc_2;
  imgdnnGetTensorDescriptor(weights_tensor, &desc_2);
  imgdnn_tensor_descriptor desc_3;
  imgdnnGetTensorDescriptor(bias_tensor, &desc_3);

  // handle weights [num_units, input_size] tensor
  /* const int order[] = { 1, 0 };
  auto isnu_weights_tensor = imgdnnNetworkTransposeOp(net,
                                                      weights_tensor,
                                                      order,
                                                      &err);*/

  fcw_tensor = imgdnnNetworkBinaryOp(
      net, reshaped_input, weights_tensor, IMGDNN_OPERATION_MATMUL, &err);

  if (bias_tensor) {
    imgdnn_tensor fcw_int_tensor =
        imgdnnNetworkCastOp(net, fcw_tensor, IMGDNN_TYPE_I32, nullptr, &err);

    imgdnn_tensor_descriptor desc_4;
    imgdnnGetTensorDescriptor(fcw_int_tensor, &desc_4);

    fcb_tensor = imgdnnNetworkBinaryOp(
        net, fcw_int_tensor, bias_tensor, IMGDNN_OPERATION_ADD, &err);
  } else {
    fcb_tensor = fcw_tensor;
  }

  imgdnn_tensor_descriptor desc;
  imgdnnGetTensorDescriptor(input_tensor, &desc);
  if (desc.type == IMGDNN_TYPE_Q_I8 || desc.type == IMGDNN_TYPE_Q_U8) {
    fcb_tensor =
        imgdnnNetworkCastOp(net, fcb_tensor, desc.type, &dst_quant_param, &err);
  }

  return fcb_tensor;
}

imgdnn_tensor ConvNetBuilder::createSoftmaxLayer(
    imgdnn_tensor input_tensor,
    float beta,
    unsigned int axis,
    imgdnn_quant_param dst_quant_param) {
  // debug
  imgdnn_tensor_descriptor desc_1;
  imgdnnGetTensorDescriptor(input_tensor, &desc_1);

  imgdnn_tensor softmax_tensor =
      imgdnnNetworkSoftmaxOp(net, input_tensor, beta, axis, &err);
  imgdnn_tensor_descriptor desc;
  imgdnnGetTensorDescriptor(input_tensor, &desc);
  if (desc.type == IMGDNN_TYPE_Q_I8 || desc.type == IMGDNN_TYPE_Q_U8) {
    softmax_tensor = imgdnnNetworkCastOp(
        net, softmax_tensor, desc.type, &dst_quant_param, &err);
  }

  imgdnn_tensor_descriptor desc_2;
  imgdnnGetTensorDescriptor(softmax_tensor, &desc_2);

  return softmax_tensor;
}

imgdnn_tensor ConvNetBuilder::createScaleLayer(imgdnn_tensor input_tensor,
                                               bool with_biasscale,
                                               const void *const scale,
                                               const void *const bias) {
  imgdnn_tensor sc_tensor;
  imgdnn_tensor scale_tensor;
  imgdnn_tensor_descriptor sc_desc;

  imgdnn_tensor broadcast2_tensor;
  imgdnn_tensor broadcast3_tensor;

  unsigned int buffer_size;

  imgdnn_tensor_descriptor in_desc;
  imgdnnGetTensorDescriptor(input_tensor, &in_desc);

  sc_desc.dimensions = 2;
  sc_desc.type = in_desc.type;
  sc_desc.size[0] = in_desc.size[0];
  sc_desc.size[1] = in_desc.size[1];

  buffer_size = imgdnnGetDescriptorSize(&sc_desc, &err);
  sc_data.push_back(reinterpret_cast<float *>(calloc(1, buffer_size)));
  memcpy(sc_data.back(), scale, buffer_size);
  scale_tensor = imgdnnNetworkFixedInput(net, &sc_desc, sc_data.back(), &err);

  broadcast2_tensor =
      imgdnnNetworkBroadcastOp(net, scale_tensor, 2, in_desc.size[2], &err);

  broadcast3_tensor = imgdnnNetworkBroadcastOp(
      net, broadcast2_tensor, 3, in_desc.size[3], &err);

  sc_tensor = imgdnnNetworkBinaryOp(
      net, input_tensor, broadcast3_tensor, IMGDNN_OPERATION_MUL, &err);

  if (with_biasscale) {
    imgdnn_tensor bsc_tensor;
    imgdnn_tensor biasscale_tensor;

    buffer_size = imgdnnGetDescriptorSize(&sc_desc, &err);
    bsc_data.push_back(reinterpret_cast<float *>(calloc(1, buffer_size)));
    memcpy(bsc_data.back(), bias, buffer_size);
    biasscale_tensor =
        imgdnnNetworkFixedInput(net, &sc_desc, bsc_data.back(), &err);

    broadcast2_tensor = imgdnnNetworkBroadcastOp(
        net, biasscale_tensor, 2, in_desc.size[2], &err);

    broadcast3_tensor = imgdnnNetworkBroadcastOp(
        net, broadcast2_tensor, 3, in_desc.size[3], &err);

    bsc_tensor = imgdnnNetworkBinaryOp(
        net, sc_tensor, broadcast3_tensor, IMGDNN_OPERATION_ADD, &err);
    return bsc_tensor;
  } else {
    return sc_tensor;
  }
}

imgdnn_network_object ConvNetBuilder::createNetworkObject(
    imgdnn_device device,
    imgdnn_context context,
    unsigned int num_inputs,
    imgdnn_tensor *inputs,
    unsigned int num_outputs,
    imgdnn_tensor *outputs) {
  const imgdnn_network_object_flags flags = 0;

  std::string options_str;
  std::string ddk_root{"/home/jasonwang/imgtools/ndk/main/"};
  std::string hwconfig =
      ddk_root + "nna-tools/config/mirage_hw_config06_23_2_6500_301.json";
  std::string mapconfig = ddk_root + "nna-tools/config/mapconfig_q8a.json";
  options_str += "-h " + hwconfig;
  options_str += " -m " + mapconfig;
  options_str += " --dump_debug_binaries enabled";

  imgdnn_network_object net_obj;
  net_obj = imgdnnCreateNetworkObject(device,
                                      context,
                                      net,
                                      num_inputs,
                                      inputs,
                                      num_outputs,
                                      outputs,
                                      flags,
                                      options_str.c_str(),
                                      &err);

  if (err == IMGDNN_SUCCESS)
    return net_obj;
  else
    return nullptr;
}

}  // namespace nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
