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

#include "lite/backends/nna/device.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace nna {

std::shared_ptr<int> Device::Build(const std::string model_name,   // NOLINT
                                   std::vector<int>& input_nodes,  // NOLINT
                                   std::vector<int>& output_nodes  // NOLINT
) {
  VLOG(3) << "[NNA] Build model";
  // Build the HiAI IR graph to the HiAI om model
#if 0
  ge::Graph ir_graph("graph");
  ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);
  ge::Model om_model("model", "model");
  om_model.SetGraph(ir_graph);
  domi::HiaiIrBuild ir_build;
  domi::ModelBufferData om_model_buf;
  if (!ir_build.CreateModelBuff(om_model, om_model_buf)) {
    LOG(WARNING) << "[NPU] CreateModelBuff failed!";
    return nullptr;
  }
  if (!ir_build.BuildIRModel(om_model, om_model_buf)) {
    LOG(WARNING) << "[NPU] BuildIRModel failed!";
    ir_build.ReleaseModelBuff(om_model_buf);
    return nullptr;
  }
#endif
  // Create a HiAI model manager client to load the HiAI om model
  std::shared_ptr<int> model_client(new int());
#if 0
  if (model_client->Init(nullptr) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient init failed)!";
    ir_build.ReleaseModelBuff(om_model_buf);
    return nullptr;
  }
  auto model_desc = std::make_shared<hiai::AiModelDescription>(
      model_name, freq_level(), framework_type(), model_type(), device_type());
  model_desc->SetModelBuffer(om_model_buf.data, om_model_buf.length);
  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_descs;
  model_descs.push_back(model_desc);
  if (model_client->Load(model_descs) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient load model failed!";
    ir_build.ReleaseModelBuff(om_model_buf);
    return nullptr;
  }
  ir_build.ReleaseModelBuff(om_model_buf);
  VLOG(3) << "[NPU] Build done";
#endif
  return model_client;
}

}  // namespace nna
}  // namespace lite
}  // namespace paddle
