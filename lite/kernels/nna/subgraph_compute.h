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

#include <memory>
#include <string>
#include <vector>
#include "imgdnn.h"  // NOLINT
#include "lite/core/kernel.h"
#include "lite/kernels/nna/bridges/graph.h"
#include "lite/kernels/npu/bridges/engine.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nna {

class SubgraphEngine : public subgraph::Engine {
 public:
  SubgraphEngine(KernelContext *ctx,
                 int block_idx,
                 cpp::BlockDesc *block_desc,
                 const std::vector<std::string> &input_names,
                 const std::vector<std::string> &output_names,
                 Scope *scope)
      : subgraph::Engine(
            ctx, block_idx, block_desc, input_names, output_names, scope) {
    network_obj_ = nullptr;
    context_ = nullptr;
    binding_ = nullptr;
  }

  ~SubgraphEngine() {
    imgdnn_err_code err;
    if (network_obj_) err = imgdnnNetworkObjectDestroy(network_obj_);
    if (context_) err = imgdnnContextDestroy(context_);
    if (binding_) err = imgdnnBindingDestroy(binding_);
    CHECK_EQ(err, IMGDNN_SUCCESS);
  }

 protected:
  int BuildDeviceProgram() override;
  int LaunchDeviceProgram() override;

  std::vector<std::string> device_inames_;
  std::vector<std::string> device_onames_;
  std::vector<imgdnn_input> device_itensors_;
  std::vector<imgdnn_output> device_otensors_;
  imgdnn_binding binding_{nullptr};
  imgdnn_context context_{nullptr};
  imgdnn_network_object network_obj_{nullptr};
};

class SubgraphCompute
    : public KernelLite<TARGET(kNNA), PRECISION(kInt8), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::SubgraphParam;

  void PrepareForRun() override;

  void Run() override;

  virtual ~SubgraphCompute() = default;

 private:
  std::unique_ptr<SubgraphEngine> engine_;
};

}  // namespace nna
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
