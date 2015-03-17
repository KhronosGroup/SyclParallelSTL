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

using namespace std::experimental::parallel;

#include <experimental/detail/sycl_iterator>
#include <string>

class multiply_by_factor {

  int m_factor;

public:
  multiply_by_factor(int factor) : m_factor(factor) {};

  int operator()(int num) const { return num * m_factor; }
};

using std::experimental::parallel::sycl::begin;
using std::experimental::parallel::sycl::end;

int main() {
  std::vector<int> v = { 3, 1, 5, 6 };
  {
    cl::sycl::buffer<int> b(v.begin(), v.end());
    b.set_final_data(v.data());

    sycl::sort(sycl::sycl_policy, begin(b), end(b));

    // Transform
    cl::sycl::default_selector h;

    {
      cl::sycl::queue q(h);
      sycl::sycl_execution_policy_named<class transform1> sepn1(q);
      sycl::transform(sepn1, begin(b), end(b), begin(b),
                      [](int num) { return num + 1; });

      sycl::sycl_execution_policy_named<class transform2> sepn2(q);
      sycl::transform(sepn2, begin(b), end(b), begin(b),
                      [](int num) { return num - 1; });

      sycl::sycl_execution_policy_named<class transform3> sepn3(q);
      sycl::transform(sepn3, begin(b), end(b), begin(b), multiply_by_factor(2));

      // Note that we can use directly STL operations :-)
      sycl::sycl_execution_policy_named<class transform4> sepn4(q);
      sycl::transform(sepn4, begin(b), end(b), begin(b), std::negate<int>());
    } // All the kernels will finish at this point */
  }   // The buffer destructor guarantees host syncrhonization
  std::sort(v.begin(), v.end());

  for (auto i = 0; i < v.size(); i++) {
    std::cout << v[i] << " , " << std::endl;
  }

  return !std::is_sorted(v.begin(), v.end());
}
