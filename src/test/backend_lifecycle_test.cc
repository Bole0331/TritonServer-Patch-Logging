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
#include "gtest/gtest.h"

#include "src/core/backend_lifecycle.h"

namespace ni = nvidia::inferenceserver;

namespace {

class BackendLifecycleTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
  }

  void TearDown() override
  { 
  }
};

TEST_F(CudaMemoryManagerTest, InitOOM)
{
  // Set to reserve too much memory
  double cc = 6.0;
  std::map<int, uint64_t> s{{0, uint64_t(1) << 40 /* 1024 GB */}};
  const ni::CudaMemoryManager::Options options{cc, s};
  auto status = ni::CudaMemoryManager::Create(options);
  EXPECT_FALSE(status.IsOk()) << "Expect creation error";
}

TEST_F(CudaMemoryManagerTest, InitSuccess)
{
  double cc = 6.0;
  std::map<int, uint64_t> s{{0, 1 << 10 /* 1024 bytes */}};
  const ni::CudaMemoryManager::Options options{cc, s};
  auto status = ni::CudaMemoryManager::Create(options);
  EXPECT_TRUE(status.IsOk()) << status.Message();
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
