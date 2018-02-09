/* Copyright (c) 2015-2018 The Khronos Group Inc.

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
#include "gmock/gmock.h"

#include <algorithm>
#include <iterator>
#include <numeric>
#include <vector>

#include <experimental/algorithm>
#include <sycl/execution_policy>
#include <sycl/helpers/sycl_iterator.hpp>

namespace parallel = std::experimental::parallel;
using namespace sycl::helpers;

class TestBufferIterator : public testing::Test {
 public:
};

TEST_F(TestBufferIterator, Basic) {
  int n_elems = 128;
  std::vector<int> in(n_elems);

  std::generate(begin(in), end(in), []() { return std::rand() % 8; });

  std::vector<int> out(in.size());
  {
    cl::sycl::queue q;

    cl::sycl::buffer<int> in_buff(in.data(), in.size());
    cl::sycl::buffer<int> out_buff(out.data(), out.size());

    sycl::sycl_execution_policy<class Increment0> snp(q);
    parallel::transform(snp, begin(in_buff), end(in_buff), begin(out_buff),
                        [](int x) { return x + 1; });
  }

  std::vector<int> expected(n_elems);

  std::transform(begin(in), end(in), begin(expected),
                 [](int x) { return x + 1; });

  EXPECT_TRUE(std::equal(begin(expected), end(expected), begin(out)));
}

TEST_F(TestBufferIterator, BeginningOnly) {
  int n_elems = 128;
  std::vector<int> in(n_elems);

  std::generate(begin(in), end(in), []() { return std::rand() % 8; });

  std::vector<int> out(in.size());
  {
    cl::sycl::queue q;

    cl::sycl::buffer<int> in_buff(in.data(), in.size());
    cl::sycl::buffer<int> out_buff(out.data(), out.size());

    sycl::sycl_execution_policy<class Increment1> snp(q);
    parallel::transform(snp, begin(in_buff), end(in_buff) - n_elems / 2,
                        begin(out_buff), [](int x) { return x + 1; });
  }

  std::vector<int> expected(n_elems);

  std::transform(begin(in), end(in) - n_elems / 2, begin(expected),
                 [](int x) { return x + 1; });

  EXPECT_TRUE(std::equal(begin(expected), end(expected), begin(out)));
}

TEST_F(TestBufferIterator, EndOnly) {
  int n_elems = 128;
  std::vector<int> in(n_elems);

  std::generate(begin(in), end(in), []() { return std::rand() % 8; });

  std::vector<int> out(in.size());
  {
    cl::sycl::queue q;

    cl::sycl::buffer<int> in_buff(in.data(), in.size());
    cl::sycl::buffer<int> out_buff(out.data(), out.size());

    sycl::sycl_execution_policy<class Increment2> snp(q);
    parallel::transform(snp, begin(in_buff) + n_elems / 2, end(in_buff),
                        begin(out_buff) + n_elems / 2, [](int x) { return x + 1; });
  }

  std::vector<int> expected(n_elems);

  std::transform(begin(in) + n_elems / 2, end(in), begin(expected) + n_elems/2,
                 [](int x) { return x + 1; });


  EXPECT_TRUE(std::equal(begin(expected), end(expected), begin(out)));
}
