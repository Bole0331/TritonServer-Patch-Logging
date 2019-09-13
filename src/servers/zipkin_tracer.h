// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <atomic>
#include <memory>
#include <string>
#include "src/core/api.pb.h"
#include "src/core/trtserver.h"

namespace nvidia { namespace inferenceserver {

//
// Singleton manager for tracing
//
class TraceManager {
 public:
  // Create the singlton trace manager that delivers trace information
  // to the specified host:port.
  static TRTSERVER_Error* Create(const std::string& hostname, uint32_t port);

  // Set the trace level and sampling rate.
  static TRTSERVER_Error* SetLevel(TRTSERVER_Trace_Level level);
  static TRTSERVER_Error* SetRate(uint32_t rate);

  // Return a trace object that should be used to collected trace
  // activities for an inference request. Return nullptr if no tracing
  // should occur.
  static TRTSERVER_Trace* SampleTrace(
      const std::string& model_name, int64_t model_version,
      const InferRequestHeader& request_header);

 private:
  TraceManager(const std::string& hostname, uint32_t port);

  // Unfortunately we need manager to be a singleton because the
  // underlying zipkin library uses singletons for the trace
  // objects... not sure why they did that...
  static std::unique_ptr<TraceManager> singleton_;

  // The trace level and sampling rate.
  TRTSERVER_Trace_Level level_;
  uint32_t rate_;

  // Atomically incrementing counter used to implement sampling rate.
  std::atomic<uint64_t> sample_;
};

}}  // namespace nvidia::inferenceserver
