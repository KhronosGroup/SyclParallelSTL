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

#include <experimental/execution_policy>
#include <experimental/algorithm>
#include <vector>
#include <iostream>
#include <algorithm>

#include <experimental/sycl_vector>

using namespace std::experimental::parallel;

#include <string>

class multiply_by_factor {

  int m_factor;

public:
  multiply_by_factor(int factor) : m_factor(factor) {};

  int operator()(int num) const { return num * m_factor; }
};

int main() {
  std::vector<int> v = { 3, 1, 5, 6 };
  std::vector<int> v2 = { 4, 5, 2 };
  sycl::sort(sycl::sycl_policy, v.begin(), v.end());

  // Transform
  cl::sycl::default_selector h;

  {
    cl::sycl::queue q(h);
    sycl::sycl_execution_policy_named<class transform1> sepn1(q);
    sycl::transform(sepn1, v2.begin(), v2.end(), v2.begin(),
                    [](int num) { return num + 1; });

    sycl::sycl_execution_policy_named<class transform2> sepn2(q);
    sycl::transform(sepn2, v2.begin(), v2.end(), v2.begin(),
                    [](int num) { return num - 1; });

    sycl::sycl_execution_policy_named<class transform3> sepn3(q);
    sycl::transform(sepn3, v2.begin(), v2.end(), v2.begin(),
                    multiply_by_factor(2));

    // Note that we can use directly STL operations :-)
    sycl::sycl_execution_policy_named<class transform4> sepn4(q);
    sycl::transform(sepn4, v2.begin(), v2.end(), v2.begin(),
                    std::negate<int>());
  } // Everything is back on the host now

  sycl::sort(sycl::sycl_policy, v2.begin(), v2.end());

  if (!std::is_sorted(v2.begin(), v2.end())) {
    std::cout << " Sequence is not sorted! " << std::endl;
    for (auto i = 0; i < v2.size(); i++) {
      std::cout << v2[i] << " , " << std::endl;
    }
  }

  return !std::is_sorted(v2.begin(), v2.end());
}
