// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include <fstream>
#include <string>
#include "src/clients/c++/perf_analyzer/client_backend/triton_local/shared_library.h"
#include "src/clients/c++/perf_analyzer/perf_utils.h"
#include "triton/core/tritonserver.h"

#include <thread>

// If TRITONSERVER error is non-OK, return the corresponding status.
#define RETURN_IF_TRITONSERVER_ERROR(E, MSG)                \
  do {                                                      \
    TRITONSERVER_Error* err__ = (E);                        \
    if (err__ != nullptr) {                                 \
      std::cerr << "error: " << (MSG) << ": "               \
                << error_code_to_string_fn_(err__) << " - " \
                << error_message_fn_(err__) << std::endl;   \
      Error newErr = cb::Error(MSG);                        \
      error_delete_fn_(err__);                              \
      return newErr;                                        \
    }                                                       \
  } while (false)

#define REPORT_TRITONSERVER_ERROR(E)                      \
  do {                                                    \
    TRITONSERVER_Error* err__ = (E);                      \
    if (err__ != nullptr) {                               \
      std::cerr << error_message_fn_(err__) << std::endl; \
      error_delete_fn_(err__);                            \
    }                                                     \
  } while (false)

namespace cb = perfanalyzer::clientbackend;
namespace perfanalyzer { namespace clientbackend {
class TritonLoader {
 public:
  TritonLoader(
      std::string library_directory, std::string memory_type,
      std::string model_repository_path)
      : library_directory_(library_directory),
        model_repository_path_(model_repository_path)
  {
    cb::Error status = LoadServerLibrary();
    // some error handling
    if (!model_repository_path.empty()) {
      model_repository_path_ = model_repository_path;
    } else {
      RETURN_ERROR("Need to specify model repository");
    }
    status = startTriton(memory_type, true);
    assert(status.IsOk());
  }

  ~TritonLoader()
  {
    FAIL_IF_ERR(
        CloseLibraryHandle(dlhandle_), "error on closing triton loader");
    ClearHandles();
  }

  cb::Error startTriton(std::string& memory_type, bool isVerbose)
  {
    if (!memory_type.empty()) {
      enforce_memory_type_ = true;
      if (memory_type.compare("system")) {
        requested_memory_type_ = TRITONSERVER_MEMORY_CPU;
      } else if (memory_type.compare("pinned")) {
        requested_memory_type_ = TRITONSERVER_MEMORY_CPU_PINNED;
      } else if (memory_type.compare("gpu")) {
        requested_memory_type_ = TRITONSERVER_MEMORY_GPU;
      } else {
        RETURN_ERROR(
            "Specify one of the following types: system, pinned or gpu");
      }
    }

    if (isVerbose) {
      verbose_level_ = 1;
    }

    // Check API version.
    uint32_t api_version_major, api_version_minor;
    REPORT_TRITONSERVER_ERROR(
        api_version_fn_(&api_version_major, &api_version_minor));
    std::cout << "api version major: " << api_version_major
              << ", minor: " << api_version_minor << std::endl;
    if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
        (TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
      RETURN_ERROR("triton server API version mismatch");
    }

    // Create the server...
    TRITONSERVER_ServerOptions* server_options = nullptr;
    RETURN_IF_TRITONSERVER_ERROR(
        options_new_fn_(&server_options), "creating server options");
    RETURN_IF_TRITONSERVER_ERROR(
        options_set_model_repo_path_fn_(
            server_options, model_repository_path_.c_str()),
        "setting model repository path");
    RETURN_IF_TRITONSERVER_ERROR(
        set_log_verbose_fn_(server_options, verbose_level_),
        "setting verbose logging level");
    RETURN_IF_TRITONSERVER_ERROR(
        set_backend_directory_fn_(
            server_options, (library_directory_ + "/backends").c_str()),
        "setting backend directory");
    RETURN_IF_TRITONSERVER_ERROR(
        set_repo_agent_directory_fn_(
            server_options, (library_directory_ + "/repoagents").c_str()),
        "setting repository agent directory");
    RETURN_IF_TRITONSERVER_ERROR(
        set_strict_model_config_fn_(server_options, true),
        "setting strict model configuration");
    double min_compute_capability = 0;
    // FIXME: Do not have GPU support right now
    RETURN_IF_TRITONSERVER_ERROR(
        set_min_supported_compute_capability_fn_(
            server_options, min_compute_capability),
        "setting minimum supported CUDA compute capability");
    RETURN_IF_TRITONSERVER_ERROR(
        server_new_fn_(&server_ptr_, server_options), "creating server");
    RETURN_IF_TRITONSERVER_ERROR(
        server_options_delete_fn_(server_options), "deleting server options");
    std::shared_ptr<TRITONSERVER_Server> shared_server(
        server_ptr_, server_delete_fn_);
    server_ = shared_server;

    // Wait until the server is both live and ready.
    size_t health_iters = 0;
    while (true) {
      bool live, ready;
      RETURN_IF_TRITONSERVER_ERROR(
          server_is_live_fn_(server_.get(), &live),
          "unable to get server liveness");
      RETURN_IF_TRITONSERVER_ERROR(
          server_is_ready_fn_(server_.get(), &ready),
          "unable to get server readiness");
      std::cout << "Server Health: live " << live << ", ready " << ready
                << std::endl;
      if (live && ready) {
        std::cout << "server is alive!" << std::endl;
        break;
      }

      if (++health_iters >= 10) {
        RETURN_ERROR("failed to find healthy inference server");
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    // Print status of the server.
    {
      TRITONSERVER_Message* server_metadata_message;
      RETURN_IF_TRITONSERVER_ERROR(
          server_metadata_fn_(server_.get(), &server_metadata_message),
          "unable to get server metadata message");
      const char* buffer;
      size_t byte_size;
      RETURN_IF_TRITONSERVER_ERROR(
          message_serialize_to_json_fn_(
              server_metadata_message, &buffer, &byte_size),
          "unable to serialize server metadata message");

      std::cout << "Server Status:" << std::endl;
      std::cout << std::string(buffer, byte_size) << std::endl;

      RETURN_IF_TRITONSERVER_ERROR(
          message_delete_fn_(server_metadata_message),
          "deleting status metadata");
    }

    return Error::Success;
  }

  // TRITONSERVER_ApiVersion
  typedef TRITONSERVER_Error* (*TritonServerApiVersionFn_t)(
      uint32_t* major, uint32_t* minor);
  // TRITONSERVER_ServerOptionsNew
  typedef TRITONSERVER_Error* (*TritonServerOptionsNewFn_t)(
      TRITONSERVER_ServerOptions** options);
  // TRITONSERVER_ServerOptionsSetModelRepositoryPath
  typedef TRITONSERVER_Error* (*TritonServerOptionSetModelRepoPathFn_t)(
      TRITONSERVER_ServerOptions* options, const char* model_repository_path);
  // TRITONSERVER_ServerOptionsSetLogVerbose
  typedef TRITONSERVER_Error* (*TritonServerSetLogVerboseFn_t)(
      TRITONSERVER_ServerOptions* options, int level);

  // TRITONSERVER_ServerOptionsSetBackendDirectory
  typedef TRITONSERVER_Error* (*TritonServerSetBackendDirFn_t)(
      TRITONSERVER_ServerOptions* options, const char* backend_dir);
  // TRITONSERVER_ServerOptionsSetRepoAgentDirectory
  typedef TRITONSERVER_Error* (*TritonServerSetRepoAgentDirFn_t)(
      TRITONSERVER_ServerOptions* options, const char* repoagent_dir);
  // TRITONSERVER_ServerOptionsSetStrictModelConfig
  typedef TRITONSERVER_Error* (*TritonServerSetStrictModelConfigFn_t)(
      TRITONSERVER_ServerOptions* options, bool strict);
  // TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability
  typedef TRITONSERVER_Error* (
      *TritonServerSetMinSupportedComputeCapabilityFn_t)(
      TRITONSERVER_ServerOptions* options, double cc);

  // TRITONSERVER_ServerNew
  typedef TRITONSERVER_Error* (*TritonServerNewFn_t)(
      TRITONSERVER_Server** server, TRITONSERVER_ServerOptions* option);
  // TRITONSERVER_ServerOptionsDelete
  typedef TRITONSERVER_Error* (*TritonServerOptionsDeleteFn_t)(
      TRITONSERVER_ServerOptions* options);
  // TRITONSERVER_ServerDelete
  typedef TRITONSERVER_Error* (*TritonServerDeleteFn_t)(
      TRITONSERVER_Server* server);
  // TRITONSERVER_ServerIsLive
  typedef TRITONSERVER_Error* (*TritonServerIsLiveFn_t)(
      TRITONSERVER_Server* server, bool* live);

  // TRITONSERVER_ServerIsReady
  typedef TRITONSERVER_Error* (*TritonServerIsReadyFn_t)(
      TRITONSERVER_Server* server, bool* ready);
  // TRITONSERVER_ServerMetadata
  typedef TRITONSERVER_Error* (*TritonServerMetadataFn_t)(
      TRITONSERVER_Server* server, TRITONSERVER_Message** server_metadata);
  // TRITONSERVER_MessageSerializeToJson
  typedef TRITONSERVER_Error* (*TritonServerMessageSerializeToJsonFn_t)(
      TRITONSERVER_Message* message, const char** base, size_t* byte_size);
  // TRITONSERVER_MessageDelete
  typedef TRITONSERVER_Error* (*TritonServerMessageDeleteFn_t)(
      TRITONSERVER_Message* message);

  // TRITONSERVER_ServerModelIsReady
  typedef TRITONSERVER_Error* (*TritonServerModelIsReadyFn_t)(
      TRITONSERVER_Server* server, const char* model_name,
      const int64_t model_version, bool* ready);
  // TRITONSERVER_ServerModelMetadata
  typedef TRITONSERVER_Error* (*TritonServerModelMetadataFn_t)(
      TRITONSERVER_Server* server, const char* model_name,
      const int64_t model_version, TRITONSERVER_Message** model_metadata);
  // TRITONSERVER_ResponseAllocatorNew
  typedef TRITONSERVER_Error* (*TritonServerResponseAllocatorNewFn_t)(
      TRITONSERVER_ResponseAllocator** allocator,
      TRITONSERVER_ResponseAllocatorAllocFn_t alloc_fn,
      TRITONSERVER_ResponseAllocatorReleaseFn_t release_fn,
      TRITONSERVER_ResponseAllocatorStartFn_t start_fn);
  // TRITONSERVER_InferenceRequestNew
  typedef TRITONSERVER_Error* (*TritonServerInferenceRequestNewFn_t)(
      TRITONSERVER_InferenceRequest** inference_request,
      TRITONSERVER_Server* server, const char* model_name,
      const int64_t model_version);

  // TRITONSERVER_InferenceRequestSetId
  typedef TRITONSERVER_Error* (*TritonServerInferenceRequestSetIdFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, const char* id);
  // TRITONSERVER_InferenceRequestSetReleaseCallback
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestSetReleaseCallbackFn_t)(
      TRITONSERVER_InferenceRequest* inference_request,
      TRITONSERVER_InferenceRequestReleaseFn_t request_release_fn,
      void* request_release_userp);
  // TRITONSERVER_InferenceRequestAddInput
  typedef TRITONSERVER_Error* (*TritonServerInferenceRequestAddInputFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, const char* name,
      const TRITONSERVER_DataType datatype, const int64_t* shape,
      uint64_t dim_count);
  // TRITONSERVER_InferenceRequestAddRequestedOutput
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestAddRequestedOutputFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, const char* name);

  // TRITONSERVER_InferenceRequestAppendInputData
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestAppendInputDataFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, const char* name,
      const void* base, size_t byte_size, TRITONSERVER_MemoryType memory_type,
      int64_t memory_type_i);
  // TRITONSERVER_InferenceRequestSetResponseCallback
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestSetResponseCallbackFn_t)(
      TRITONSERVER_InferenceRequest* inference_request,
      TRITONSERVER_ResponseAllocator* response_allocator,
      void* response_allocator_userp,
      TRITONSERVER_InferenceResponseCompleteFn_t response_fn,
      void* response_userp);
  // TRITONSERVER_ServerInferAsync
  typedef TRITONSERVER_Error* (*TritonServerInferAsyncFn_t)(
      TRITONSERVER_Server* server,
      TRITONSERVER_InferenceRequest* inference_request,
      TRITONSERVER_InferenceTrace* trace);
  // TRITONSERVER_InferenceResponseError
  typedef TRITONSERVER_Error* (*TritonServerInferenceResponseErrorFn_t)(
      TRITONSERVER_InferenceResponse* inference_response);

  // TRITONSERVER_InferenceResponseDelete
  typedef TRITONSERVER_Error* (*TritonServerInferenceResponseDeleteFn_t)(
      TRITONSERVER_InferenceResponse* inference_response);
  // TRITONSERVER_InferenceRequestRemoveAllInputData
  typedef TRITONSERVER_Error* (
      *TritonServerInferenceRequestRemoveAllInputDataFn_t)(
      TRITONSERVER_InferenceRequest* inference_request, const char* name);
  // TRITONSERVER_ResponseAllocatorDelete
  typedef TRITONSERVER_Error* (*TritonServerResponseAllocatorDeleteFn_t)(
      TRITONSERVER_ResponseAllocator* allocator);
  // TRITONSERVER_ErrorNew
  typedef TRITONSERVER_Error* (*TritonServerErrorNewFn_t)(
      TRITONSERVER_Error_Code code, const char* msg);

  // TRITONSERVER_MemoryTypeString
  typedef const char* (*TritonServerMemoryTypeStringFn_t)(
      TRITONSERVER_MemoryType memtype);
  // TRITONSERVER_InferenceResponseOutputCount
  typedef TRITONSERVER_Error* (*TritonServerInferenceResponseOutputCountFn_t)(
      TRITONSERVER_InferenceResponse* inference_response, uint32_t* count);
  // TRITONSERVER_DataTypeString
  typedef const char* (*TritonServerDataTypeStringFn_t)(
      TRITONSERVER_DataType datatype);
  // TRITONSERVER_ErrorMessage
  typedef const char* (*TritonServerErrorMessageFn_t)(
      TRITONSERVER_Error* error);
  // TRITONSERVER_ErrorDelete
  typedef void (*TritonServerErrorDeleteFn_t)(TRITONSERVER_Error* error);
  // TRITONSERVER_ErrorCodeString
  typedef const char* (*TritonServerErrorCodeToStringFn_t)(
      TRITONSERVER_Error* error);

 private:
  /// Load all tritonserver.h functions onto triton_loader
  /// internal handles
  cb::Error LoadServerLibrary()
  {
    std::string full_path = library_directory_ + SERVER_LIBRARY_PATH;
    RETURN_IF_ERROR(FileExists(full_path));
    FAIL_IF_ERR(
        OpenLibraryHandle(full_path, &dlhandle_),
        "shared library loading library:" + full_path);

    TritonServerApiVersionFn_t apifn;
    TritonServerOptionsNewFn_t onfn;
    TritonServerOptionSetModelRepoPathFn_t rpfn;
    TritonServerSetLogVerboseFn_t slvfn;

    TritonServerSetBackendDirFn_t sbdfn;
    TritonServerSetRepoAgentDirFn_t srdfn;
    TritonServerSetStrictModelConfigFn_t ssmcfn;
    TritonServerSetMinSupportedComputeCapabilityFn_t smsccfn;

    TritonServerNewFn_t snfn;
    TritonServerOptionsDeleteFn_t odfn;
    TritonServerDeleteFn_t sdfn;
    TritonServerIsLiveFn_t ilfn;

    TritonServerIsReadyFn_t irfn;
    TritonServerMetadataFn_t smfn;
    TritonServerMessageSerializeToJsonFn_t stjfn;
    TritonServerMessageDeleteFn_t mdfn;

    TritonServerModelIsReadyFn_t mirfn;
    TritonServerModelMetadataFn_t mmfn;
    TritonServerResponseAllocatorNewFn_t ranfn;
    TritonServerInferenceRequestNewFn_t irnfn;

    TritonServerInferenceRequestSetIdFn_t irsifn;
    TritonServerInferenceRequestSetReleaseCallbackFn_t irsrcfn;
    TritonServerInferenceRequestAddInputFn_t iraifn;
    TritonServerInferenceRequestAddRequestedOutputFn_t irarofn;

    TritonServerInferenceRequestAppendInputDataFn_t iraidfn;
    TritonServerInferenceRequestSetResponseCallbackFn_t irsrescfn;
    TritonServerInferAsyncFn_t iafn;
    TritonServerInferenceResponseErrorFn_t irefn;

    TritonServerInferenceResponseDeleteFn_t irdfn;
    TritonServerInferenceRequestRemoveAllInputDataFn_t irraidfn;
    TritonServerResponseAllocatorDeleteFn_t iradfn;
    TritonServerErrorNewFn_t enfn;

    TritonServerMemoryTypeStringFn_t mtsfn;
    TritonServerInferenceResponseOutputCountFn_t irocfn;
    TritonServerDataTypeStringFn_t dtsfn;
    TritonServerErrorMessageFn_t emfn;
    TritonServerErrorDeleteFn_t edfn;
    TritonServerErrorCodeToStringFn_t ectsfn;

    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ApiVersion", true /* optional */,
        reinterpret_cast<void**>(&apifn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerOptionsNew", true /* optional */,
        reinterpret_cast<void**>(&onfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerOptionsSetModelRepositoryPath",
        true /* optional */, reinterpret_cast<void**>(&rpfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerOptionsSetLogVerbose",
        true /* optional */, reinterpret_cast<void**>(&slvfn)));

    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerOptionsSetBackendDirectory",
        true /* optional */, reinterpret_cast<void**>(&sbdfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerOptionsSetRepoAgentDirectory",
        true /* optional */, reinterpret_cast<void**>(&srdfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerOptionsSetStrictModelConfig",
        true /* optional */, reinterpret_cast<void**>(&ssmcfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability",
        true /* optional */, reinterpret_cast<void**>(&smsccfn)));

    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerNew", true /* optional */,
        reinterpret_cast<void**>(&snfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerOptionsDelete", true /* optional */,
        reinterpret_cast<void**>(&odfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerDelete", true /* optional */,
        reinterpret_cast<void**>(&sdfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerIsLive", true /* optional */,
        reinterpret_cast<void**>(&ilfn)));

    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerIsReady", true /* optional */,
        reinterpret_cast<void**>(&irfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerMetadata", true /* optional */,
        reinterpret_cast<void**>(&smfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_MessageSerializeToJson", true /* optional */,
        reinterpret_cast<void**>(&stjfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_MessageDelete", true /* optional */,
        reinterpret_cast<void**>(&mdfn)));

    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerModelIsReady", true /* optional */,
        reinterpret_cast<void**>(&mirfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerModelMetadata", true /* optional */,
        reinterpret_cast<void**>(&mmfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ResponseAllocatorNew", true /* optional */,
        reinterpret_cast<void**>(&ranfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_InferenceRequestNew", true /* optional */,
        reinterpret_cast<void**>(&irnfn)));

    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_InferenceRequestSetId", true /* optional */,
        reinterpret_cast<void**>(&irsifn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_InferenceRequestSetReleaseCallback",
        true /* optional */, reinterpret_cast<void**>(&irsrcfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_InferenceRequestAddInput", true /* optional */,
        reinterpret_cast<void**>(&iraifn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_InferenceRequestAddRequestedOutput",
        true /* optional */, reinterpret_cast<void**>(&irarofn)));

    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_InferenceRequestAppendInputData",
        true /* optional */, reinterpret_cast<void**>(&iraidfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_InferenceRequestSetResponseCallback",
        true /* optional */, reinterpret_cast<void**>(&irsrescfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ServerInferAsync", true /* optional */,
        reinterpret_cast<void**>(&iafn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_InferenceResponseError", true /* optional */,
        reinterpret_cast<void**>(&irefn)));

    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_InferenceResponseDelete", true /* optional */,
        reinterpret_cast<void**>(&irdfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_InferenceRequestRemoveAllInputData",
        true /* optional */, reinterpret_cast<void**>(&irraidfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ResponseAllocatorDelete", true /* optional */,
        reinterpret_cast<void**>(&iradfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ErrorNew", true /* optional */,
        reinterpret_cast<void**>(&enfn)));

    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_MemoryTypeString", true /* optional */,
        reinterpret_cast<void**>(&mtsfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_InferenceResponseOutputCount",
        true /* optional */, reinterpret_cast<void**>(&irocfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_DataTypeString", true /* optional */,
        reinterpret_cast<void**>(&dtsfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ErrorMessage", true /* optional */,
        reinterpret_cast<void**>(&emfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ErrorDelete", true /* optional */,
        reinterpret_cast<void**>(&edfn)));
    RETURN_IF_ERROR(GetEntrypoint(
        dlhandle_, "TRITONSERVER_ErrorCodeString", true /* optional */,
        reinterpret_cast<void**>(&ectsfn)));

    api_version_fn_ = apifn;
    options_new_fn_ = onfn;
    options_set_model_repo_path_fn_ = rpfn;
    set_log_verbose_fn_ = slvfn;

    set_backend_directory_fn_ = sbdfn;
    set_repo_agent_directory_fn_ = srdfn;
    set_strict_model_config_fn_ = ssmcfn;
    set_min_supported_compute_capability_fn_ = smsccfn;

    server_new_fn_ = snfn;
    server_options_delete_fn_ = odfn;
    server_delete_fn_ = sdfn;
    server_is_live_fn_ = ilfn;

    server_is_ready_fn_ = irfn;
    server_metadata_fn_ = smfn;
    message_serialize_to_json_fn_ = stjfn;
    message_delete_fn_ = mdfn;

    model_is_ready_fn_ = mirfn;
    model_metadata_fn_ = mmfn;
    response_allocator_new_fn_ = ranfn;
    inference_request_new_fn_ = irnfn;

    inference_request_set_id_fn_ = irsifn;
    inference_request_set_release_callback_fn_ = irsrcfn;
    inference_request_add_input_fn_ = iraifn;
    inference_request_add_requested_output_fn_ = irarofn;

    inference_request_append_input_data_fn_ = iraidfn;
    inference_request_set_response_callback_fn_ = irsrescfn;
    infer_async_fn_ = iafn;
    inference_response_error_fn_ = irefn;

    inference_response_delete_fn_ = irdfn;
    inference_request_remove_all_input_data_fn_ = irraidfn;
    response_allocator_delete_fn_ = iradfn;
    error_new_fn_ = enfn;

    memory_type_string_fn_ = mtsfn;
    inference_response_output_count_fn_ = irocfn;
    data_type_string_fn_ = dtsfn;
    error_message_fn_ = emfn;
    error_delete_fn_ = edfn;
    error_code_to_string_fn_ = ectsfn;

    return Error::Success;
  }

  void ClearHandles()
  {
    dlhandle_ = nullptr;

    api_version_fn_ = nullptr;
    options_new_fn_ = nullptr;
    options_set_model_repo_path_fn_ = nullptr;
    set_log_verbose_fn_ = nullptr;

    set_backend_directory_fn_ = nullptr;
    set_repo_agent_directory_fn_ = nullptr;
    set_strict_model_config_fn_ = nullptr;
    set_min_supported_compute_capability_fn_ = nullptr;

    server_new_fn_ = nullptr;
    server_options_delete_fn_ = nullptr;
    server_delete_fn_ = nullptr;
    server_is_live_fn_ = nullptr;

    server_is_ready_fn_ = nullptr;
    server_metadata_fn_ = nullptr;
    message_serialize_to_json_fn_ = nullptr;
    message_delete_fn_ = nullptr;

    model_is_ready_fn_ = nullptr;
    model_metadata_fn_ = nullptr;
    response_allocator_new_fn_ = nullptr;
    inference_request_new_fn_ = nullptr;

    inference_request_set_id_fn_ = nullptr;
    inference_request_set_release_callback_fn_ = nullptr;
    inference_request_add_input_fn_ = nullptr;
    inference_request_add_requested_output_fn_ = nullptr;

    inference_request_append_input_data_fn_ = nullptr;
    inference_request_set_response_callback_fn_ = nullptr;
    infer_async_fn_ = nullptr;
    inference_response_error_fn_ = nullptr;

    inference_response_delete_fn_ = nullptr;
    inference_request_remove_all_input_data_fn_ = nullptr;
    response_allocator_delete_fn_ = nullptr;
    error_new_fn_ = nullptr;

    memory_type_string_fn_ = nullptr;
    inference_response_output_count_fn_ = nullptr;
    data_type_string_fn_ = nullptr;
    error_message_fn_ = nullptr;
    error_delete_fn_ = nullptr;
    error_code_to_string_fn_ = nullptr;

    options_ = nullptr;
    server_ptr_ = nullptr;
  }

  /// Check if file exists in the current directory
  /// \param filepath Path of library to check
  /// \return perfanalyzer::clientbackend::Error
  Error FileExists(std::string& filepath)
  {
    std::ifstream ifile;
    ifile.open(filepath);
    if (!ifile) {
      return Error("unable to find local Triton library: " + filepath);
    } else {
      return Error::Success;
    }
  }

  void* dlhandle_;
  TritonServerApiVersionFn_t api_version_fn_;
  TritonServerOptionsNewFn_t options_new_fn_;
  TritonServerOptionSetModelRepoPathFn_t options_set_model_repo_path_fn_;
  TritonServerSetLogVerboseFn_t set_log_verbose_fn_;

  TritonServerSetBackendDirFn_t set_backend_directory_fn_;
  TritonServerSetRepoAgentDirFn_t set_repo_agent_directory_fn_;
  TritonServerSetStrictModelConfigFn_t set_strict_model_config_fn_;
  TritonServerSetMinSupportedComputeCapabilityFn_t
      set_min_supported_compute_capability_fn_;

  TritonServerNewFn_t server_new_fn_;
  TritonServerOptionsDeleteFn_t server_options_delete_fn_;
  TritonServerDeleteFn_t server_delete_fn_;
  TritonServerIsLiveFn_t server_is_live_fn_;

  TritonServerIsReadyFn_t server_is_ready_fn_;
  TritonServerMetadataFn_t server_metadata_fn_;
  TritonServerMessageSerializeToJsonFn_t message_serialize_to_json_fn_;
  TritonServerMessageDeleteFn_t message_delete_fn_;

  TritonServerModelIsReadyFn_t model_is_ready_fn_;
  TritonServerModelMetadataFn_t model_metadata_fn_;
  TritonServerResponseAllocatorNewFn_t response_allocator_new_fn_;
  TritonServerInferenceRequestNewFn_t inference_request_new_fn_;

  TritonServerInferenceRequestSetIdFn_t inference_request_set_id_fn_;
  TritonServerInferenceRequestSetReleaseCallbackFn_t
      inference_request_set_release_callback_fn_;
  TritonServerInferenceRequestAddInputFn_t inference_request_add_input_fn_;
  TritonServerInferenceRequestAddRequestedOutputFn_t
      inference_request_add_requested_output_fn_;

  TritonServerInferenceRequestAppendInputDataFn_t
      inference_request_append_input_data_fn_;
  TritonServerInferenceRequestSetResponseCallbackFn_t
      inference_request_set_response_callback_fn_;
  TritonServerInferAsyncFn_t infer_async_fn_;
  TritonServerInferenceResponseErrorFn_t inference_response_error_fn_;

  TritonServerInferenceResponseDeleteFn_t inference_response_delete_fn_;
  TritonServerInferenceRequestRemoveAllInputDataFn_t
      inference_request_remove_all_input_data_fn_;
  TritonServerResponseAllocatorDeleteFn_t response_allocator_delete_fn_;
  TritonServerErrorNewFn_t error_new_fn_;

  TritonServerMemoryTypeStringFn_t memory_type_string_fn_;
  TritonServerInferenceResponseOutputCountFn_t
      inference_response_output_count_fn_;
  TritonServerDataTypeStringFn_t data_type_string_fn_;
  TritonServerErrorMessageFn_t error_message_fn_;
  TritonServerErrorDeleteFn_t error_delete_fn_;
  TritonServerErrorCodeToStringFn_t error_code_to_string_fn_;

  TRITONSERVER_ServerOptions* options_;
  TRITONSERVER_Server* server_ptr_;
  std::shared_ptr<TRITONSERVER_Server> server_;
  const std::string library_directory_;
  const std::string SERVER_LIBRARY_PATH = "/lib/libtritonserver.so";
  int verbose_level_ = 0;
  bool enforce_memory_type_ = false;
  std::string model_repository_path_;
  TRITONSERVER_memorytype_enum requested_memory_type_ = TRITONSERVER_MEMORY_CPU;
};


}}  // namespace perfanalyzer::clientbackend