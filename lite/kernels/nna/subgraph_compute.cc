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

#include "lite/kernels/nna/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <limits>
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/kernels/nna/bridges/graph.h"
#include "lite/kernels/nna/bridges/paddle_use_bridges.h"
#include "lite/kernels/nna/bridges/utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace nna {

int SubgraphEngine::BuildDeviceProgram() {
  int status = 0;
  // Convert all of ops and their input vars and weights and added into the NNA
  // IMG IR graph
  subgraph::nna::Graph graph{&imgdnn_mgr_};
  const auto& bridges = subgraph::Registry::Instance();
  for (auto& inst : origin_program_) {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists(op_type, TARGET(kNNA))) {
      return subgraph::FAILED;
    }
    auto kernel = inst.kernel();
    status |=
        bridges.Select(op_type, TARGET(kNNA))(reinterpret_cast<void*>(&graph),
                                              const_cast<OpLite*>(op),
                                              const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return subgraph::FAILED;
    }
  }

  // Collect the valid input and output nodes in the IMGDNN IR graph and update
  // the input and output names
  device_inames_.clear();
  std::vector<imgdnn_tensor> device_inodes;
  for (auto& input_name : input_names_) {
    if (graph.Has(input_name)) {
      device_inodes.push_back(graph.Get(input_name)->data());
      device_inames_.push_back(input_name);
    } else {
      LOG(WARNING) << "[NNA] Input node " << input_name
                   << " is ignored because it does not exist.";
    }
  }

  device_onames_.clear();
  std::vector<imgdnn_tensor> device_onodes;
  for (auto& output_name : output_names_) {
    if (graph.Has(output_name)) {
      device_onodes.push_back(graph.Get(output_name)->data());
      device_onames_.push_back(output_name);
    } else {
      LOG(WARNING) << "[NNA] Output node " << output_name
                   << " is ignored because it does not exist.";
    }
  }
  CHECK(!device_inames_.empty())
      << "[NNA] No input nodes found for building NNA model";
  CHECK(!device_onames_.empty())
      << "[NNA] No output nodes found for building NNA model";

  imgdnn_mgr_.createNetworkObject(device_inodes.size(),
                                  device_inodes.data(),
                                  device_onodes.size(),
                                  device_onodes.data());

  // inputs
  unsigned int num_inputs, num_outputs;
  imgdnn_mgr_.getNetworkObjectInputs(std::numeric_limits<unsigned int>::max(),
      nullptr, &num_inputs);
  CHECK_EQ(num_inputs, device_inames_.size());
  origin_idims_.resize(num_inputs);
  origin_itensors_.resize(num_inputs);
  device_itensors_.resize(num_inputs);
  imgdnn_mgr_.getNetworkObjectInputs(num_inputs, device_itensors_.data(),
      nullptr);
  for (int i = 0; i < num_inputs; i++) {
    auto node = graph.Get(device_inames_[i]);
    auto type = node->type();
    auto layout = node->layout();
    origin_itensors_[i] = scope_->FindMutableTensor(device_inames_[i]);
    CHECK(origin_itensors_[i]);
    origin_idims_[i] = origin_itensors_[i]->dims();
    VLOG(3) << "[NNA] Inputs[" << i << "] name: " << device_inames_[i]
            << " type: " << type << " layout: " << DataLayoutToStr(layout);
  }

  // outputs
  imgdnn_mgr_.getNetworkObjectOutputs(std::numeric_limits<unsigned int>::max(),
      nullptr, &num_outputs);
  CHECK_EQ(num_outputs, device_onames_.size());
  origin_odims_.resize(num_outputs);
  origin_otensors_.resize(num_outputs);
  device_otensors_.resize(num_outputs);
  imgdnn_mgr_.getNetworkObjectOutputs(num_outputs, device_otensors_.data(),
      nullptr);
  for (int i = 0; i < num_outputs; i++) {
    auto node = graph.Get(device_onames_[i]);
    auto type = node->type();
    auto layout = node->layout();
    origin_otensors_[i] = scope_->FindMutableTensor(device_onames_[i]);
    CHECK(origin_otensors_[i]);
    origin_odims_[i] = origin_otensors_[i]->dims();
    VLOG(3) << "[NNA] Outputs[" << i << "] name: " << device_onames_[i]
            << " type: " << type << " layout: " << DataLayoutToStr(layout);
    // Prepare the device output tensors
    switch (type) {
      case IMGDNN_TYPE_F32:
        origin_otensors_[i]->mutable_data<float>();
        break;
      case IMGDNN_TYPE_Q_I8:
      case IMGDNN_TYPE_Q_U8:
        origin_otensors_[i]->mutable_data<int8_t>();
        break;
      case IMGDNN_TYPE_I16:
        origin_otensors_[i]->mutable_data<int16_t>();
        break;
      case IMGDNN_TYPE_I32:
        origin_otensors_[i]->mutable_data<int32_t>();
        break;
      default:
        LOG(FATAL) << "[NNA] " << device_onames_[i]
                   << " can't mutable data with precision type " << type;
        break;
    }
  }

  return status;
}

int SubgraphEngine::LaunchDeviceProgram() {
  // using the data of origin input tensors as the buffer
  // of imgdnn_input tensors
  imgdnn_input in;
  imgdnn_tensor_descriptor in_desc;
  size_t in_size;
  void* in_buf;
  imgdnn_memory in_mem;
  Tensor input_temp;
  for (size_t i = 0; i < device_itensors_.size(); i++) {
    input_temp.Resize({origin_idims_[i]});
    uint8_t* input_data = input_temp.mutable_data<uint8_t>();
    int8_t* input_raw_data =
        reinterpret_cast<int8_t*>(origin_itensors_[i]->raw_data());
    for (int j = 0; j < origin_itensors_[i]->data_size(); j++) {
      input_data[j] = (uint8_t)(input_raw_data[j] + 128);
    }

    in = device_itensors_[i];
    in_desc = imgdnn_mgr_.getInputDescriptor(in);
    in_size = imgdnn_mgr_.getDescriptorSize(&in_desc);
    in_buf = static_cast<void*>(input_data);
    CHECK_EQ(in_size, origin_itensors_[i]->memory_size());
    in_mem = imgdnn_mgr_.importMemory(in_buf, in_size);
    imgdnn_mgr_.addBindingInput(in, in_mem);
  }

  // set output
  imgdnn_output out;
  imgdnn_tensor_descriptor out_desc;
  size_t out_size;
  imgdnn_memory out_mem;
  std::vector<imgdnn_memory> out_mems;
  std::vector<size_t> out_mem_sizes;
  for (size_t i = 0; i < device_otensors_.size(); i++) {
    out = device_otensors_[i];
    out_desc = imgdnn_mgr_.getOutputDescriptor(out);
    out_size = imgdnn_mgr_.getDescriptorSize(&out_desc);
    CHECK_EQ(out_size, origin_otensors_[i]->memory_size());
    out_mem = imgdnn_mgr_.allocateMemory(out_size);
    out_mems.push_back(out_mem);
    out_mem_sizes.push_back(out_size);
    imgdnn_mgr_.addBindingOutput(out, out_mem);
  }

  // Run the img model by name
  imgdnn_mgr_.executeNetworkObject(true, 0, nullptr, nullptr);

  // Copy the data of output tensor to the buffer of origin output tensors
  for (size_t i = 0; i < out_mems.size(); i++) {
    uint8_t* data = static_cast<uint8_t*>(
        imgdnn_mgr_.lockMemory(out_mems[i], IMGDNN_LOCK_ACCESS_READ_ONLY));

    int8_t* output_data = origin_otensors_[i]->mutable_data<int8_t>();
    for (size_t j = 0; j < out_mem_sizes[i]; j++) {
      output_data[j] = data[j] - 128;
    }
    imgdnn_mgr_.unlockMemory(out_mems[i]);
    imgdnn_mgr_.destroyMemory(out_mems[i]);
  }

  return 0;
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(ctx_.get(),
                                   param.sub_block_idx,
                                   param.sub_block_desc,
                                   param.input_data_names,
                                   param.output_data_names,
                                   param.scope));
  CHECK(engine_);
  engine_->Build();
}

void SubgraphCompute::Run() {
  CHECK(engine_);
  engine_->Launch();
}

}  // namespace nna
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kNNA,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::nna::SubgraphCompute,
                     def)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .Finalize();
