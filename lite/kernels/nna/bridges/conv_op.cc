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

#include "lite/operators/conv_op.h"
#include "lite/kernels/nna/bridges/graph.h"
#include "lite/kernels/nna/bridges/registry.h"
#include "lite/kernels/nna/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nna {

int ConvConverter(void *ctx, OpLite *op, KernelBase *kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph *>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NNA] Converting " << op_type << "... ";

  // Get input and output vars and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();

  auto filter_name = op_info->Input("Filter").front();
  auto filter = scope->FindMutableTensor(filter_name);
  auto filter_dims = filter->dims();

  auto output_name = op_info->Output("Output").front();
  auto output = scope->FindMutableTensor(output_name);
  auto output_dims = output->dims();

  auto bs = input_dims[0];
  auto ic = input_dims[1];
  auto oc = filter_dims[0];
  CHECK_EQ(input_dims.size(), 4L);
  CHECK_EQ(output_dims.size(), 4L);
  CHECK_EQ(filter_dims.size(), 4L);
  CHECK_EQ(output_dims[0], bs);
  CHECK_EQ(output_dims[1], oc);
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto groups = op_info->GetAttr<int>("groups");
  auto dilations = op_info->GetAttr<std::vector<int>>("dilations");
  bool with_act =
      op_info->HasAttr("with_act") && op_info->GetAttr<bool>("with_act");
  std::string act_type =
      with_act ? op_info->GetAttr<std::string>("act_type") : "";

  CHECK_EQ(strides.size(), 2L);
  CHECK_EQ(dilations.size(), 2L);

  // for quantization
  bool enable_int8 = false;
  float input_scale = 1.0;
  float output_scale = 1.0;
  std::vector<float> weight_scale;
  TensorInfo qnt;

  if (op_info->HasAttr("enable_int8")) {
    enable_int8 = op_info->GetAttr<bool>("enable_int8");
    input_scale = op_info->GetAttr<float>("input_scale");
    output_scale = op_info->GetAttr<float>("output_scale");
    weight_scale = op_info->GetAttr<std::vector<float>>("weight_scale");
  }


  // Input node
  std::shared_ptr<Node> input_node = nullptr;
  imgdnn_tensor in_tensor;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
    in_tensor = input_node->data();
  } else {
    TensorInfoReset(&qnt);
    if (enable_int8)
      qnt.type = IMGDNN_TYPE_Q_U8;
    else
      qnt.type = IMGDNN_TYPE_F32;
    qnt.scales.push_back(input_scale);
    qnt.zero_points.push_back(128);
    input_node = graph->Add(input_name, *input, qnt, Node::Role::kInput);
    in_tensor = input_node->data();
  }

  if (paddings.size() == 2L) {
    for (size_t i = 0; i < strides.size(); ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L)
      << "[NNA] Paddings size should be the same or twice as the input size.";

  std::string padding_algorithm("");
  if (op_info->HasAttr("padding_algorithm")) {
    padding_algorithm = op_info->GetAttr<std::string>("padding_algorithm");
  }
  operators::UpdatePaddingAndDilation(&paddings,
                                      &dilations,
                                      strides,
                                      padding_algorithm,
                                      input_dims,
                                      filter_dims);

  // Check depthwise mode, and decide whether use ConvolutionDepthwise Op
  bool is_depthwise_mode = (ic == groups && oc == groups && groups != 1);

  // Filter node
  std::shared_ptr<Node> filter_node = nullptr;
  imgdnn_tensor filter_tensor;
  bool per_channel = isScalesPerChannel(weight_scale);
  TensorInfoReset(&qnt);
  uint8_t *weights_u8 = graph->GetBufromPool(filter_dims.production());
  if (enable_int8) {
    char *weight_src = static_cast<char *>(filter->raw_data());

    qnt.type = IMGDNN_TYPE_Q_U8;
    if (per_channel) {
      qnt.scales.assign(weight_scale.begin(), weight_scale.end());
      qnt.zero_points.assign(weight_scale.size(), 128);
      qnt.count = oc;
      qnt.axis = 1;
    } else {
      qnt.scales.push_back(weight_scale.at(0));
      qnt.zero_points.push_back(128);
    }
    for (int i = 0; i < filter_dims.production(); i++) {
      weights_u8[i] = static_cast<uint8_t>(weight_src[i] + 128);
    }

    filter_node = graph->Add(filter_name,
                             weights_u8,
                             filter_dims.Vectorize(),
                             qnt,
                             Node::Role::kConst);
    filter_tensor = filter_node->data();
  } else {
    qnt.type = IMGDNN_TYPE_F32;
    filter_node = graph->Add(filter_name, *filter, qnt, Node::Role::kConst);
  }

  // Add bias node if exists bias
  // Supports the bias nodes with the following dimensions
  // 0: {oc}
  // 1: {1, oc, oh, ow}
  // 2: {n, oc, oh, ow}
  std::shared_ptr<Node> bias_node = NULL;
  imgdnn_tensor bias_tensor = nullptr;
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_name = op_info->Input("Bias").front();
    if (graph->Has(bias_name)) {
      bias_node = graph->Get(bias_name);
    } else {
      auto bias = scope->FindMutableTensor(bias_name);
      auto bias_dims = bias->dims();
      auto bias_data_size = bias_dims.production();
      auto output_data_size = output_dims.production();
      std::vector<int64_t> bias_shape;
      if (bias_data_size == oc) {
        // 0: {oc}
        bias_shape = {1, oc, 1, 1};
      } else if (bias_data_size == output_data_size / bs) {
        // 1: {1, oc, oh, ow}
        bias_shape = {1, output_dims[1], output_dims[2], output_dims[3]};
      } else if (bias_data_size == output_data_size) {
        // 2: {n, oc, oh, ow}
        bias_shape = output_dims.Vectorize();
      } else {
        LOG(WARNING)
            << "[NNA] Bias dimension " << bias_dims
            << " isn't supported in conv2d Op when output dimension is "
            << output_dims;
        return FAILED;
      }

      TensorInfoReset(&qnt);
      std::vector<int64_t> shapes{1, oc};
      auto bias_data = bias->data<float, float>();
      if (enable_int8) {
        qnt.type = IMGDNN_TYPE_I32;
        if (per_channel) {
          qnt.scales.resize(bias_data_size);
          for (int i = 0; i < bias_data_size; i++)
            qnt.scales[i] = input_scale * weight_scale[i];
          qnt.zero_points.assign(bias_data_size, 0);
          qnt.count = 2;
          qnt.axis = 1;
        } else {
          qnt.scales.push_back(input_scale * weight_scale[0]);
          qnt.zero_points.push_back(0);
        }

        int quant_bits = 32;
        auto dtype_max = static_cast<int>((1 << (quant_bits - 1)) - 1);
        auto dtype_min = static_cast<int>(0 - dtype_max);

        int32_t *bias_qnt_data = reinterpret_cast<int32_t *>(
            graph->GetBufromPool(bias_dims.production() * sizeof(int32_t)));
        for (int i = 0; i < bias_data_size; i++) {
          float current_scale = per_channel ? qnt.scales[i] : qnt.scales[0];
          bias_qnt_data[i] =
              std::min(std::max(static_cast<int>(bias_data[i] / current_scale),
                                dtype_min),
                       dtype_max);
        }

        bias_node = graph->Add(
            bias_name, bias_qnt_data, shapes, qnt, Node::Role::kConst);
      } else {
        qnt.type = IMGDNN_TYPE_F32;
        std::vector<float> bias_float_data(bias_data,
                                           bias_data + bias_data_size);
        bias_node = graph->Add(
            bias_name, bias_float_data.data(), shapes, qnt, Node::Role::kConst);
      }
      bias_tensor = bias_node->data();
    }
  }

  unsigned int img_stride[2] = {(unsigned int)strides[0],
                                (unsigned int)strides[1]};
  unsigned int pad_to_begin[2] = {(unsigned int)paddings[0],
                                  (unsigned int)paddings[2]};  // top,left
  unsigned int pad_to_end[2] = {(unsigned int)paddings[1],
                                (unsigned int)paddings[3]};  // bottom,right
  unsigned int img_dilation[2] = {(unsigned int)dilations[0],
                                  (unsigned int)dilations[1]};

  imgdnn_quant_param output_quant_param;
  output_quant_param.scale = output_scale;
  output_quant_param.zero_point = 128;

  imgdnn_tensor conv_out = graph->GetBuilder()->createConvolutionLayer(
                                                          in_tensor,
                                                          filter_tensor,
                                                          bias_tensor,
                                                          output_quant_param,
                                                          img_stride,
                                                          pad_to_begin,
                                                          pad_to_end,
                                                          img_dilation,
                                                          is_depthwise_mode);

  imgdnn_tensor_descriptor desc;
  imgdnn_err_code err = imgdnnGetTensorDescriptor(conv_out, &desc);
  CHECK(err == IMGDNN_SUCCESS) << "fail get tensor description(CONV)";

  graph->Add(output_name, conv_out, desc.type);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(conv2d,
                         kNNA,
                         paddle::lite::subgraph::nna::ConvConverter);

REGISTER_SUBGRAPH_BRIDGE(depthwise_conv2d,
                         kNNA,
                         paddle::lite::subgraph::nna::ConvConverter);