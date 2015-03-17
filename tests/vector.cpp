/* Copyright (c) 2015, Codeplay Software Ltd.
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */  
#include "gmock/gmock.h"

#include <vector>
#include <algorithm>

#include <experimental/detail/sycl_iterator>

class SyclHostIteratorTest : public testing::Test {
public:
};

using std::experimental::parallel::sycl::begin;
using std::experimental::parallel::sycl::end;

TEST_F(SyclHostIteratorTest, TestIteratorsOnHostAccessor) {
  std::vector<float> v = { 1, 3, 5, 7, 9 };
  // TODO(Ruyman) : Workaround for #5327
  // cl::sycl::buffer<float> sv(cl::sycl::range<1>(v.size()));
  cl::sycl::buffer<float> sv(v.begin(), v.end());

  ASSERT_EQ(sv.get_count(), v.size());

  {
    // A host vector is just a vector that contains a host_accessor
    // The data of the vector is the host accessor
    auto hostAcc = sv.get_access<cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::host_buffer>();

    auto vI = v.begin();
    int count = 0;

    for (; vI != v.end(); vI++, count++) {
      hostAcc[count] = *vI;
      ASSERT_EQ(*vI, hostAcc[count]);
    }

    vI = v.begin();
    count = 0;

    for (auto i = begin(hostAcc); i != end(hostAcc); i++, vI++) {
      // DEBUG
      // printf("[%d] %g == %g  \n", count, *i, hostAcc[count]);
      EXPECT_EQ(*vI, *i);
      ASSERT_LT(count++, sv.get_size());
    }
  }
}

TEST_F(SyclHostIteratorTest, TestUsingStlAlgorithm) {
  std::vector<float> v = { 1, 3, 5, 7, 9 };
  // TODO(Ruyman) : Workaround for #5327
  // cl::sycl::buffer<float> sv(cl::sycl::range<1>(v.size()));
  cl::sycl::buffer<float> sv(v.begin(), v.end());

  auto hostAcc = sv.get_access<cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::host_buffer>();

  std::transform(begin(hostAcc), end(hostAcc), begin(hostAcc),
                 [=](float e) { return e * 2; });

  auto vI = v.begin();
  int count = 0;

  for (auto i = begin(hostAcc); i != end(hostAcc); i++, vI++) {
    // DEBUG
    // printf("[%d] %g == %g (%p == %p) \n",
    //          count, *i,
    //          hostAcc[count], &hostAcc[0], &(*i));
    ASSERT_EQ(*(vI) * 2, *i);
    count++;
  }
}
