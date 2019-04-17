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

#include <algorithm>
#include <iostream>
#include <vector>

#include <experimental/algorithm>
#include <sycl/execution_policy>

using namespace std::experimental::parallel;

sycl::sycl_execution_policy<> sycl_policy;

/* This sample tests the updated multi-kernel bitonic sort implementation.
 * We use a sycl buffer to perform all operations on
 * the device.
 * Note that for the moment the sycl variants of the algorithm
 *   are on the sycl namespace and not in std::experimental.
 */
template <typename T> inline T init_num(T num, int max) {
  if (std::is_integral<T>::value) {
    return num;
  } else {
    return num / max;
  }
}

template <typename T> struct typename_as_str {};

template <> struct typename_as_str<unsigned char> {
  static constexpr const char *name_ = "uchar";
};

template <> struct typename_as_str<char> {
  static constexpr const char *name_ = "char";
};

template <> struct typename_as_str<unsigned short> {
  static constexpr const char *name_ = "ushort";
};

template <> struct typename_as_str<short> {
  static constexpr const char *name_ = "short";
};

template <> struct typename_as_str<unsigned int> {
  static constexpr const char *name_ = "uint";
};

template <> struct typename_as_str<int> {
  static constexpr const char *name_ = "int";
};


template <> struct typename_as_str<unsigned long> {
  static constexpr const char *name_ = "ulong";
};

template <> struct typename_as_str<long> {
  static constexpr const char *name_ = "long";
};

template <> struct typename_as_str<float> {
  static constexpr const char *name_ = "float";
};

template <> struct typename_as_str<double> {
  static constexpr const char *name_ = "double";
};

template <typename T>
bool test(const int minInputSizeLog2 = 3, const int maxInputSizeLog2 = 5) {

  bool sorted = true;

  std::cout << __FUNCTION__ << "<" << typename_as_str<T>::name_ << ">"
            << std::endl;

  for (int i = minInputSizeLog2; i <= maxInputSizeLog2; ++i) {

    std::vector<T> v;
    v.resize(1 << i);

    std::cout << "in : ";
    for (int j = 0; j < v.size(); ++j) {
      v[j] = init_num(static_cast<T>((v.size() - 1) - j), v.size());
      std::cout << (v[j]) << (j == v.size() - 1 ? "" : ", ");
    }
    std::cout << std::endl;

    std::experimental::parallel::sort(sycl_policy, v.begin(), v.end());

    std::cout << "out : ";
    for (size_t j = 0; j < v.size(); j++) {
      std::cout << (v[j]) << (j == v.size() - 1 ? "" : ", ");
    }
    std::cout << std::endl;

    sorted = sorted && std::is_sorted(v.begin(), v.end());
    if (!sorted) {
      std::cout << "failed!" << std::endl;
      break;
    }
  }
  if(sorted)
  {
    std::cout << "success!" << std::endl;
  }
  return sorted;
}

int main() {
  bool sorted = true;

  sorted = sorted && test<unsigned char>();
  sorted = sorted && test<char>();
  sorted = sorted && test<unsigned short>(); 
  sorted = sorted && test<short>();
  sorted = sorted && test<unsigned int>(); 
  sorted = sorted && test<int>();
  sorted = sorted && test<unsigned long>(); 
  sorted = sorted && test<long>(); 
  sorted = sorted && test<float>(); 
  sorted = sorted && test<double>(); 

  return !sorted;
}
