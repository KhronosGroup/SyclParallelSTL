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
/* vim: set filetype=cpp foldmethod=indent: */
#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_SORT__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_SORT__

#include <type_traits>
#include <typeinfo>

#include <experimental/execution_policy>
// Detail header
#include <experimental/detail/sycl_buffers.hpp>

#include <algorithm>
#include <iostream>

namespace std {
namespace experimental {
namespace parallel {
namespace sycl {
namespace detail {

template <typename T>
inline bool isPowerOfTwo(T num) {
  return (num != 0) && !(num & (num - 1));
}

template <typename T>
using sycl_rw_acc = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                                       cl::sycl::access::target::global_buffer>;

template <typename T>
void sort_swap (T &lhs, T &rhs) {
  auto temp = rhs;
  rhs = lhs;
  lhs = temp;
}

template <typename T> 
class sort_kernel_sequential
{
  sycl_rw_acc<T> a_;
  size_t vS_;

  public:
    sort_kernel_sequential(sycl_rw_acc<T> a, size_t vectorSize) : a_(a), vS_(vectorSize) {};

    // Simple sequential sort
    void operator()() {
      for (int i = 0; i < vS_; i++) {
        for (int j = 1; j < vS_; j++) {
          if (a_[j - 1] > a_[j]) {
            sort_swap<T>(a_[j-1], a_[j]); 
          }
        }
      }
    }
}; // class sort_kernel

template<typename T>
void sequential_sort(cl::sycl::queue q, cl::sycl::buffer<T, 1> buf, 
    size_t vectorSize) {

  auto f = [buf, vectorSize](cl::sycl::handler &h) mutable {
    auto a = buf.template get_access<cl::sycl::access::mode::read_write>(h);
    h.single_task(sort_kernel_sequential<T>(a, vectorSize));
  };
  q.submit(f);
}

template<typename T>
class sort_kernel_bitonic;

template<typename T>
void bitonic_sort(cl::sycl::queue q, cl::sycl::buffer<T, 1> buf, 
    size_t vectorSize) {
  int numStages = 0;
  // 2^numStages should be equal to length
  // i.e number of times you halve the lenght to get 1 should be numStages
  for ( int tmp = vectorSize; tmp > 1; tmp >>= 1 ) {
    ++numStages;
  }
  cl::sycl::range<1> r{ vectorSize / 2 };
  for (int stage = 0; stage < numStages; ++stage) {
    // Every stage has stage + 1 passes
    for (int passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
      auto f = [=](cl::sycl::handler &h) mutable {
        auto a = buf.template get_access<cl::sycl::access::mode::read_write>(h);
        h.parallel_for< sort_kernel_bitonic<T> >(cl::sycl::nd_range<1>{r}, 
            [a, stage, passOfStage](cl::sycl::item<1> it) {
              int sortIncreasing = 1;
              cl::sycl::id<1> id = it.get();
              int threadId = id.get(0);

              int pairDistance = 1 << (stage - passOfStage);
              int blockWidth = 2 * pairDistance;

              int leftId = (threadId % pairDistance) 
                             + (threadId / pairDistance) * blockWidth;
              int rightId = leftId + pairDistance;

              T leftElement = a[leftId];
              T rightElement = a[rightId];

              int sameDirectionBlockWidth = 1 << stage;

              if ((threadId / sameDirectionBlockWidth) % 2 == 1) {
                sortIncreasing = 1 - sortIncreasing;
              }

              T greater;
              T lesser;

              if (leftElement > rightElement) {
                greater = leftElement;
                lesser = rightElement;
              } else {
                greater = rightElement;
                lesser = leftElement;
              }

              a[leftId] = sortIncreasing?lesser:greater;
              a[rightId] = sortIncreasing?greater:lesser;
            });
      };  // command group functor
      q.submit(f);
    }  // passStage
  } // stage
}  // bitonic_sort 

}  // namespace detail 
}  // namespace sycl
} // namespace parallel
} // namespace experimental
} // namespace std

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_SORT__
