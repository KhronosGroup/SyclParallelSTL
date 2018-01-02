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

#include <algorithm>
#include <utility>
#include <vector>

#include <experimental/algorithm>
#include <sycl/execution_policy>

using namespace std::experimental::parallel;

class MismatchAlgorithm : public testing::Test {
 public:
};

TEST_F(MismatchAlgorithm, TestMismatchEqual) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(v.begin(), v.end(), v.begin());

  sycl::sycl_execution_policy<class EqualAlgorithmEqual> snp{};
  auto actual = std::experimental::parallel::mismatch(
      snp, v.begin(), v.end(), v.begin(), v.end(),
      [](int a, int b) { return a == b; });

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchEqualFirstSmaller) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(v.begin(), v.end() - 2, v.begin());

  sycl::sycl_execution_policy<class EqualAlgorithmEqualFirstSmaller> snp{};
  auto actual = std::experimental::parallel::mismatch(
      snp, v.begin(), v.end() - 2, v.begin(), v.end(),
      [](int a, int b) { return a == b; });

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchEqualSecondSmaller) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(v.begin(), v.end() - 2, v.begin());
  std::swap(std::get<0>(expected), std::get<1>(expected));

  sycl::sycl_execution_policy<class EqualAlgorithmEqualSecondSmaller> snp{};
  auto actual = std::experimental::parallel::mismatch(
      snp, v.begin(), v.end(), v.begin(), v.end() - 2,
      [](int a, int b) { return a == b; });

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchNotEqual) {
  std::vector<int> v1{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> v2{0, 1, 2, 3, 6, 5, 6, 7};

  auto expected = std::mismatch(v1.begin(), v1.end(), v2.begin());

  sycl::sycl_execution_policy<class EqualAlgorithmNotEqual> snp{};
  auto actual = std::experimental::parallel::mismatch(
      snp, v1.begin(), v1.end(), v2.begin(), v2.end(),
      [](int a, int b) { return a == b; });

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchFirstEmpty) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(v.begin(), v.begin(), v.begin());

  sycl::sycl_execution_policy<class EqualAlgorithmFirstEmpty> snp{};
  auto actual = std::experimental::parallel::mismatch(
      snp, v.begin(), v.begin(), v.begin(), v.end(),
      [](int a, int b) { return a == b; });

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchSecondEmpty) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(v.begin(), v.begin(), v.begin());
  std::swap(std::get<0>(expected), std::get<1>(expected));

  sycl::sycl_execution_policy<class EqualAlgorithmSecondEmpty> snp{};
  auto actual = std::experimental::parallel::mismatch(
      snp, v.begin(), v.end(), v.begin(), v.begin(),
      [](int a, int b) { return a == b; });

  EXPECT_EQ(actual, expected);
}

TEST_F(MismatchAlgorithm, TestMismatchBothEmpty) {
  std::vector<int> v{0, 1, 2, 3, 4, 5, 6, 7};

  auto expected = std::mismatch(v.begin(), v.begin(), v.begin());

  sycl::sycl_execution_policy<class EqualAlgorithmBothEmpty> snp{};
  auto actual = std::experimental::parallel::mismatch(
      snp, v.begin(), v.begin(), v.begin(), v.begin(),
      [](int a, int b) { return a == b; });

  EXPECT_EQ(actual, expected);
}
