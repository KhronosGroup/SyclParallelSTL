/* Copyright (c) 2015 The Khronos Group Inc.

  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and/or associated documentation files (the
  "Materials"), to deal in the Materials without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Materials, and to
  permit persons to whom the Materials are furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Materials.

  MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
  KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
  SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
     https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
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
