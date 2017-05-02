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
#include "gmock/gmock.h"

#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>

#include <sycl/execution_policy>
#include <experimental/algorithm>

using namespace std::experimental::parallel;

struct TransformReduceAlgorithm : public testing::Test {};

TEST_F(TransformReduceAlgorithm, TestSyclTransformReduce) {
  std::vector<int> v = {2, 1, 3, 4};

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TransformAlgorithm> snp(q);
  int result = transform_reduce(snp, v.begin(), v.end(),
                                [=](int val) { return val * 2; }, 0,
                                [=](int v1, int v2) { return v1 + v2; });

  EXPECT_TRUE(20 == result);
}

TEST_F(TransformReduceAlgorithm, TestSyclTransformReduce2) {
  std::vector<int> v;
  int n = 128;

  for (int i = 0; i < n; i++) {
    v.push_back(1);
  }

  cl::sycl::queue q;
  sycl::sycl_execution_policy<class TransformReduce2Algorithm> snp(q);
  int ressycl = transform_reduce(snp, v.begin(), v.end(),
                                 [=](int val) { return val * 2; }, 0,
                                 [=](int v1, int v2) { return v1 + v2; });

  EXPECT_TRUE( (2*128) == ressycl);
}
