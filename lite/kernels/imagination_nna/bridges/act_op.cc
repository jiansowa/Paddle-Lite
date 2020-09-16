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

#include "lite/kernels/imagination_nna/bridges/graph.h"
#include "lite/kernels/imagination_nna/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace imagination_nna {

// template <typename ActType>
int ActConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NNA] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    // x_node = graph->Add(x_name, *x);
    LOG(WARNING) << "ActConverter:x_node not in graph";
  }

  imgdnn_tensor relu_output = graph->GetBuilder()->createReLULayer(
      x_node->data(), true, 0.0, false, 0.0, 0.0);

  imgdnn_tensor_descriptor desc;
  imgdnn_err_code err = imgdnnGetTensorDescriptor(relu_output, &desc);
  CHECK(err == IMGDNN_SUCCESS) << "fail get tensor description(RELU)";

  graph->Add(out_name, relu_output, desc.type);

  return SUCCESS;
}

}  // namespace imagination_nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
#if 0
REGISTER_SUBGRAPH_BRIDGE(
    sigmoid,
    kImaginationNNA,
    paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Activation>);
#endif
REGISTER_SUBGRAPH_BRIDGE(relu,
                         kImaginationNNA,
                         paddle::lite::subgraph::imagination_nna::ActConverter);
#if 0
REGISTER_SUBGRAPH_BRIDGE(
    tanh, kImaginationNNA, paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Activation>);
REGISTER_SUBGRAPH_BRIDGE(
    relu_clipped,
    kImaginationNNA,
    paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Activation>);
REGISTER_SUBGRAPH_BRIDGE(
    relu6, kImaginationNNA, paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Activation>);
REGISTER_SUBGRAPH_BRIDGE(
    leaky_relu,
    kImaginationNNA,
    paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Activation>);
REGISTER_SUBGRAPH_BRIDGE(
    abs, kImaginationNNA, paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Activation>);
REGISTER_SUBGRAPH_BRIDGE(
    softsign,
    kImaginationNNA,
    paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Activation>);
REGISTER_SUBGRAPH_BRIDGE(
    softplus,
    kImaginationNNA,
    paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Activation>);
REGISTER_SUBGRAPH_BRIDGE(
    hard_sigmoid,
    kImaginationNNA,
    paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Activation>);

REGISTER_SUBGRAPH_BRIDGE(
    log, kImaginationNNA, paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Log>);
REGISTER_SUBGRAPH_BRIDGE(
    square, kImaginationNNA, paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Square>);
REGISTER_SUBGRAPH_BRIDGE(
    sqrt, kImaginationNNA, paddle::lite::subgraph::imagination_nna::ActConverter<ge::op::Sqrt>);
#endif