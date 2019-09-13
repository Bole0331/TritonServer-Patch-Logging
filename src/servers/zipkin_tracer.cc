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

#include "src/servers/zipkin_tracer.h"

#include <cppkin.h>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/servers/common.h"

namespace nvidia { namespace inferenceserver {

std::unique_ptr<TraceManager> TraceManager::singleton_;

TRTSERVER_Error*
TraceManager::Create(const std::string& hostname, uint32_t port)
{
  // If trace object is already created then configure has already
  // been called. Can only configure once since the zipkin library we
  // are using doesn't allow reconfiguration.
  if (singleton_ != nullptr) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_ALREADY_EXISTS, "tracing is already configured");
  }

  if (hostname.empty()) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        "trace configuration requires a non-empty host name");
  }

  if (port == 0) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_INVALID_ARG,
        "trace configuration requires a non-zero port");
  }

  LOG_INFO << "Configure trace: " << hostname << ":" << port;

  singleton_.reset(new TraceManager(hostname, port));
  return nullptr;  // success
}

TraceManager::TraceManager(const std::string& hostname, uint32_t port)
    : level_(TRTSERVER_TRACE_LEVEL_DISABLED), rate_(1000), sample_(1)
{
  auto transportType = cppkin::TransportType::Http;
  auto encodingType = cppkin::EncodingType::Json;

  std::string service_name("TRTIS");

  cppkin::CppkinParams cppkin_params;
  cppkin_params.AddParam(cppkin::ConfigTags::HOST_ADDRESS, hostname);
  cppkin_params.AddParam(cppkin::ConfigTags::PORT, port);
  cppkin_params.AddParam(cppkin::ConfigTags::SERVICE_NAME, service_name);
  cppkin_params.AddParam(cppkin::ConfigTags::SAMPLE_COUNT, 1);
  cppkin_params.AddParam(
      cppkin::ConfigTags::TRANSPORT_TYPE,
      cppkin::TransportType(transportType).ToString());
  cppkin_params.AddParam(
      cppkin::ConfigTags::ENCODING_TYPE,
      cppkin::EncodingType(encodingType).ToString());

  cppkin::Init(cppkin_params);
}

TRTSERVER_Error*
TraceManager::SetLevel(TRTSERVER_Trace_Level level)
{
  if (singleton_ == nullptr) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_UNAVAILABLE, "tracing is not yet configured");
  }

  // We don't bother with a mutex here since this is the only writer.
  singleton_->level_ = level;

  LOG_INFO << "Setting trace level: " << level;

  return nullptr;  // success
}

TRTSERVER_Error*
TraceManager::SetRate(uint32_t rate)
{
  if (singleton_ == nullptr) {
    return TRTSERVER_ErrorNew(
        TRTSERVER_ERROR_UNAVAILABLE, "tracing is not yet configured");
  }

  // We don't bother with a mutex here since this is the only writer.
  singleton_->rate_ = rate;

  LOG_INFO << "Setting trace rate: " << rate;

  return nullptr;  // success
}

namespace {

void
TraceActivity(
    TRTSERVER_Trace* trace, TRTSERVER_Trace_Activity activity,
    uint64_t timestamp_ns, void* userp)
{
  cppkin::Trace* zipkin_trace = reinterpret_cast<cppkin::Trace*>(userp);

  const char* activity_name = "<unknown>";
  switch (activity) {
    case TRTSERVER_TRACE_REQUEST_START:
      activity_name = "request start";
      break;
    case TRTSERVER_TRACE_QUEUE_START:
      activity_name = "queue start";
      break;
    case TRTSERVER_TRACE_COMPUTE_START:
      activity_name = "compute start";
      break;
    case TRTSERVER_TRACE_COMPUTE_END:
      activity_name = "compute end";
      break;
    case TRTSERVER_TRACE_REQUEST_END:
      activity_name = "request end";
      break;
  }

  zipkin_trace->AddAnnotation(activity_name, timestamp_ns / 1000);
  if (activity == TRTSERVER_TRACE_REQUEST_END) {
  // FIXME lifecycle
    zipkin_trace->Submit();
    delete zipkin_trace;
  }
}

}  // namespace

TRTSERVER_Trace*
TraceManager::SampleTrace(
    const std::string& model_name, int64_t model_version,
    const InferRequestHeader& request_header)
{
  // If tracing isn't configured or if the sample rate hasn't been
  // reached then don't trace.
  if (singleton_ == nullptr) {
    return nullptr;
  }

  uint64_t s = singleton_->sample_.fetch_add(1);
  if ((s % singleton_->rate_) != 0) {
    return nullptr;
  }

  // FIXME get trace start time so we can normalize?
  cppkin::Trace* zipkin_trace = new cppkin::Trace("inference");
  zipkin_trace->AddTag("model_name", model_name.c_str());
  zipkin_trace->AddTag("model_version", std::to_string(model_version).c_str());
  zipkin_trace->AddTag("id", std::to_string(request_header.id()).c_str());

  TRTSERVER_Trace* trace = nullptr;
  LOG_IF_ERR(
      TRTSERVER_TraceNew(
          &trace, singleton_->level_, TraceActivity,
          reinterpret_cast<void*>(zipkin_trace)),
      "creating trace object");

  return trace;
}

}}  // namespace nvidia::inferenceserver
