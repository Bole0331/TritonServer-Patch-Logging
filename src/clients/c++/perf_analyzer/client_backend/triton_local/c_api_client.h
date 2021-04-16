// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

/// \file

#include <queue>
#include "src/clients/c++/library/common.h"
#include "src/clients/c++/library/ipc.h"
#include "src/core/model_config.pb.h"

namespace nvidia { namespace inferenceserver { namespace client {

/// The key-value map type to be included in the request
/// metadata
typedef std::map<std::string, std::string> Headers;


//==============================================================================
/// An InferenceServerGrpcClient object is used to perform any kind of
/// communication with the InferenceServer using gRPC protocol.  Most
/// of the methods are thread-safe except Infer, AsyncInfer, StartStream
/// StopStream and AsyncStreamInfer. Calling these functions from different
/// threads will cause undefined behavior.
///
/// \code
///   std::unique_ptr<InferenceServerGrpcClient> client;
///   InferenceServerGrpcClient::Create(&client, "localhost:8001");
///   bool live;
///   client->IsServerLive(&live);
///   ...
///   ...
/// \endcode
///
class InferenceServerCapi : public InferenceServerClient {
 public:
  ~InferenceServerCapi();

  /// Create a client that can be used to communicate with the server.
  /// \param client Returns a new CAPI object
  /// \param library_path 
  /// \return Error object indicating success or failure.
  static Error Create(
      std::unique_ptr<InferenceServerCapi>* client,
      const std::string& library_path) {
          TritonLoader(library_path);
      }

  /// Contact the inference server and get its liveness.
  /// \param live Returns whether the server is live or not.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \return Error object indicating success or failure of the request.
  Error IsServerLive(
      bool* live, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get its readiness.
  /// \param ready Returns whether the server is ready or not.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error IsServerReady(
      bool* ready, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the readiness of specified model.
  /// \param ready Returns whether the specified model is ready or not.
  /// \param model_name The name of the model to check for readiness.
  /// \param model_version The version of the model to check for readiness.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error IsModelReady(
      bool* ready, const std::string& model_name,
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get its metadata.
  /// \param server_metadata Returns JSON representation of the
  /// metadata as a string.
  /// \param headers Optional map specifying additional HTTP headers to
  /// include in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ServerMetadata(
      std::string* server_metadata, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the metadata of specified model.
  /// \param model_metadata Returns JSON representation of model
  /// metadata as a string.
  /// \param model_name The name of the model to get metadata.
  /// \param model_version The version of the model to get metadata.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelMetadata(
      std::string* model_metadata, const std::string& model_name,
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the configuration of specified model.
  /// \param model_config Returns JSON representation of model
  /// configuration as a string.
  /// \param model_name The name of the model to get configuration.
  /// \param model_version The version of the model to get configuration.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelConfig(
      std::string* model_config, const std::string& model_name,
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the index of model repository
  /// contents.
  /// \param repository_index Returns JSON representation of the
  /// repository index as a string.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelRepositoryIndex(
      std::string* repository_index, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the inference server to load or reload specified model.
  /// \param model_name The name of the model to be loaded or reloaded.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error LoadModel(
      const std::string& model_name, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Request the inference server to unload specified model.
  /// \param model_name The name of the model to be unloaded.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error UnloadModel(
      const std::string& model_name, const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Contact the inference server and get the inference statistics for the
  /// specified model name and version.
  /// \param infer_stat Returns the JSON representation of the
  /// inference statistics as a string.
  /// \param model_name The name of the model to get inference statistics. The
  /// default value is an empty string which means statistics of all models will
  /// be returned in the response.
  /// \param model_version The version of the model to get inference statistics.
  /// The default value is an empty string which means then the server will
  /// choose a version based on the model and internal policy.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the request.
  Error ModelInferenceStatistics(
      std::string* infer_stat, const std::string& model_name = "",
      const std::string& model_version = "", const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

  /// Run  inference on server.
  /// \param result Returns the result of inference.
  /// \param options The options for inference request.
  /// \param inputs The vector of InferInput describing the model inputs.
  /// \param outputs The vector of InferRequestedOutput describing how the
  /// output must be returned. If not provided then all the outputs in the
  /// model config will be returned as default settings.
  /// \param headers Optional map specifying additional HTTP headers to include
  /// in request.
  /// \param query_params Optional map specifying parameters that must be
  /// included with URL query.
  /// \return Error object indicating success or failure of the
  /// request.
  Error Infer(
      InferResult** result, const InferOptions& options,
      const std::vector<InferInput*>& inputs,
      const std::vector<const InferRequestedOutput*>& outputs =
          std::vector<const InferRequestedOutput*>(),
      const Headers& headers = Headers(),
      const Parameters& query_params = Parameters());

 private:
  InferenceServerCapi(
      const std::string& library_path, bool verbose);
  // The server library path
  const std::string library_path_;


};


}}}  // namespace nvidia::inferenceserver::client
