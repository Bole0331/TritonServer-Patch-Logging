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

#include "triton/core/tritonserver.h"
#include "src/clients/c++/perf_analyzer/client_backend/triton_local/shared_library.h"
#include "src/clients/c++/perf_analyzer/perf_utils.h"
#include <fstream>
#include <string>
namespace perfanalyzer { namespace clientbackend {
class TritonLoader {
public:
typedef TRITONSERVER_Error* (*TritonServerOptionsNewFn_t)(
    TRITONSERVER_ServerOptions** options);
typedef TRITONSERVER_Error*(*TritonServerOptionSetModelRepoPathFn_t)(
    TRITONSERVER_ServerOptions* options, const char* model_repository_path);
typedef TRITONSERVER_Error* (*TritonServerServerNewFn_t)(
    TRITONSERVER_Server** server, TRITONSERVER_ServerOptions* options);
typedef TRITONSERVER_Error* (*TritonServerServerDeleteFn_t)(
    TRITONSERVER_Server* server);

TritonLoader(std::string library_directory)
 : library_directory_(library_directory) {   
   auto status = LoadServerLibrary();

   assert(status.IsOk());
    // TRITONSERVER_ServerOptions* server_options = nullptr;
    // TRITONSERVER_ServerOptionsNew(&server_options);
    // TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options, "test");
    // TRITONSERVER_Server* server_ptr = nullptr;
// TRITONSERVER_ApiVersion(&api_version_major, &api_version_minor),"getting Triton API version");
//      TRITONSERVER_ServerOptionsNew(&server_options);
//TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options, model_repository_path.c_str())
// TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level)
// TRITONSERVER_ServerOptionsSetBackendDirectory(server_options, "/opt/tritonserver/backends")
// TRITONSERVER_ServerOptionsSetRepoAgentDirectory(server_options, "/opt/tritonserver/repoagents")
// TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true)
// TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(server_options, min_compute_capability)
//TRITONSERVER_ServerNew(&server_ptr, server_options)
// TRITONSERVER_ServerOptionsDelete(server_options)
// TRITONSERVER_ServerIsLive(server.get(), &live)
// TRITONSERVER_ServerIsReady(server.get(), &ready)
// TRITONSERVER_ServerMetadata(server.get(), &server_metadata_message)
// TRITONSERVER_MessageSerializeToJson(server_metadata_message, &buffer, &byte_size)
// TRITONSERVER_MessageDelete(server_metadata_message)
// TRITONSERVER_ServerModelIsReady(server.get(), model_name.c_str(), 1, &is_ready)
}

Error LoadServerLibrary() {
    std::string full_path = library_directory_ + SERVER_LIBRARY_PATH;
    RETURN_IF_ERROR(FileExists(full_path));
    FAIL_IF_ERR(OpenLibraryHandle(full_path, &dlhandle_), "shared library loading library:" + full_path);

    TritonServerOptionsNewFn_t onfn;
    TritonServerOptionSetModelRepoPathFn_t rpfn;
    TritonServerServerNewFn_t snfn;
    TritonServerServerDeleteFn_t sdfn;

    RETURN_IF_ERROR(GetEntrypoint(dlhandle_, "TRITONSERVER_ServerOptionsNew", true /* optional */,
      reinterpret_cast<void**>(&onfn)));
    RETURN_IF_ERROR(GetEntrypoint(dlhandle_, "TRITONSERVER_ServerOptionsSetModelRepositoryPath", true /* optional */,
      reinterpret_cast<void**>(&rpfn)));
    RETURN_IF_ERROR(GetEntrypoint(dlhandle_, "TRITONSERVER_ServerNew", true /* optional */,
      reinterpret_cast<void**>(&snfn)));
    RETURN_IF_ERROR(GetEntrypoint(dlhandle_, "TRITONSERVER_ServerDelete", true /* optional */,
      reinterpret_cast<void**>(&sdfn)));

    options_new_fn_ = onfn;
    set_model_path_fn_ = rpfn;
    server_new_fn_ = snfn;
    server_delete_fn_ = sdfn;
    return Error::Success;
}

~TritonLoader() {
  FAIL_IF_ERR(CloseLibraryHandle(dlhandle_), "error on closing triton loader");
  ClearHandles();
}
private:
 void ClearHandles() {
     dlhandle_ = nullptr;
     options_new_fn_ = nullptr;
     set_model_path_fn_ = nullptr;
     server_new_fn_ = nullptr;
     server_delete_fn_ = nullptr;
     options_ = nullptr;
 }

  Error FileExists(std::string& filepath) {
    std::ifstream ifile;
    ifile.open(filepath);
    if (!ifile) {
      return Error("unable to find library: " + filepath);
    } else {
      return Error::Success;
    }
  }

void* dlhandle_;
TritonServerOptionsNewFn_t options_new_fn_;
TritonServerOptionSetModelRepoPathFn_t set_model_path_fn_;
TritonServerServerNewFn_t server_new_fn_;
TritonServerServerDeleteFn_t server_delete_fn_;
TRITONSERVER_ServerOptions* options_;
TRITONSERVER_Server* server_;
const std::string library_directory_;
const std::string SERVER_LIBRARY_PATH = "/lib/libtritonserver.so";

};



    
}}