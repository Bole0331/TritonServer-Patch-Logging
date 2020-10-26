// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#pragma once

#include <functional>
#include <map>
#include <mutex>
#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

class InferenceServer;
class InferenceBackend;

class BackendLifeCycle {
 public:
  static Status Create(
      InferenceServer* server, const double min_compute_capability,
      const BackendConfigMap& backend_config_map,
      const BackendCmdlineConfigMap& backend_cmdline_config_map,
      std::unique_ptr<BackendLifeCycle>* life_cycle);

  ~BackendLifeCycle() { map_.clear(); }

  // Start loading model backend with specified versions asynchronously.
  // The previously loaded versions will be unloaded once the async loads are
  // completed, even if the versions failed to load.
  Status AsyncLoad(
      const std::string& repository_path, const std::string& model_name,
      const std::set<int64_t>& versions,
      const inference::ModelConfig& model_config,
      std::function<void(const std::map<int64_t, ModelReadyState>&)>
          OnComplete);

  // Start unloading all versions of the model backend asynchronously.
  Status AsyncUnload(const std::string& model_name);

  // Get specified model version's backend. Latest ready version will
  // be retrieved if 'version' is -1. Return error if the version specified is
  // not found or it is not ready.
  Status GetInferenceBackend(
      const std::string& model_name, const int64_t version,
      std::shared_ptr<InferenceBackend>* backend);

  // Get the ModelStateMap representation of the live backends. A backend is
  // live if at least one of the versions is not unknown nor unavailable.
  // If 'strict_readiness' is true, a backend is only live if
  // at least one of the versions is ready.
  const ModelStateMap LiveBackendStates(bool strict_readiness = false);

  // Get the ModelStateMap representation of the backends.
  const ModelStateMap BackendStates();

  // Get the VersionStateMap representation of the specified model.
  const VersionStateMap VersionStates(const std::string& model_name);

  // Get the state of a specific model version.
  Status ModelState(
      const std::string& model_name, const int64_t model_version,
      ModelReadyState* state);

 private:
  struct BackendInfo {
    BackendInfo(
        const std::string& repository_path, const ModelReadyState state,
        const inference::ModelConfig& model_config)
        : repository_path_(repository_path),
          platform_(GetPlatform(model_config.platform())), state_(state),
          last_action_time_ns_(0), last_unload_time_ns_(0),
          model_config_(model_config)
    {
    }

    std::string repository_path_;
    Platform platform_;

    std::recursive_mutex mtx_;
    ModelReadyState state_;
    std::string state_reason_;

    // The timestamp of the latest load / unload requested, it will be used to
    // determine whether the backend state should be updated based on
    // current load / unload result.
    uint64_t last_action_time_ns_;
    // Timestamp used when unload is completed to determine whether the unload
    // is the latest action taken
    uint64_t last_unload_time_ns_;
    // callback function that will be triggered when there is no next action
    std::function<void()> OnComplete_;
    inference::ModelConfig model_config_;

    std::shared_ptr<InferenceBackend> backend_;
  };

  BackendLifeCycle(const double min_compute_capability)
      : min_compute_capability_(min_compute_capability)
  {
  }

  void CreateInferenceBackend(
      const std::string& model_name, const int64_t version,
      BackendInfo* backend_info);

  const double min_compute_capability_;

  using VersionMap = std::map<int64_t, std::unique_ptr<BackendInfo>>;
  using BackendMap = std::map<std::string, VersionMap>;
  BackendMap map_;
  std::mutex map_mtx_;

#ifdef TRITON_ENABLE_CAFFE2
  std::unique_ptr<NetDefBackendFactory> netdef_factory_;
#endif  // TRITON_ENABLE_CAFFE2
#ifdef TRITON_ENABLE_CUSTOM
  std::unique_ptr<TritonBackendFactory> triton_backend_factory_;
  std::unique_ptr<CustomBackendFactory> custom_factory_;
#endif  // TRITON_ENABLE_CUSTOM
#ifdef TRITON_ENABLE_TENSORRT
  std::unique_ptr<PlanBackendFactory> plan_factory_;
#endif  // TRITON_ENABLE_TENSORRT
#ifdef TRITON_ENABLE_PYTORCH
  std::unique_ptr<LibTorchBackendFactory> libtorch_factory_;
#endif  // TRITON_ENABLE_PYTORCH
#ifdef TRITON_ENABLE_ENSEMBLE
  std::unique_ptr<EnsembleBackendFactory> ensemble_factory_;
#endif  // TRITON_ENABLE_ENSEMBLE
};

}}  // namespace nvidia::inferenceserver
