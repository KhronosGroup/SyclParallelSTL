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
#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_TRANSFORM__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_TRANSFORM__

#include <type_traits>
#include <algorithm>
#include <iostream>

// Detail header
#include <sycl/helpers/sycl_buffers.hpp>

namespace sycl {
namespace impl {

/** transform sycl implementation
 * @brief Function that takes a Unary Operator and applies to the given range
 * @param sep : Execution Policy
 * @param b   : Start of the range
 * @param e   : End of the range
 * @param out : Output iterator
 * @param op  : Unary Operator
 * @return  An iterator pointing to the last element
 */
template <class ExecutionPolicy, class Iterator, class OutputIterator,
          class UnaryOperation>
OutputIterator transform(ExecutionPolicy &sep, Iterator b, Iterator e,
                         OutputIterator out, UnaryOperation op) {
  {
    cl::sycl::queue q(sep.get_queue());
    typedef typename std::iterator_traits<Iterator>::value_type type_;
    auto bufI = sycl::helpers::make_const_buffer(b, e);
    auto bufO = sycl::helpers::make_buffer(out, out + bufI.get_count());
    auto vectorSize = bufI.get_count();
    auto f = [vectorSize, &bufI, &bufO, op](cl::sycl::handler &h) mutable {
      const size_t local = 128;
      cl::sycl::nd_range<3> r{
          cl::sycl::range<3>{std::max(vectorSize, local), 1, 1},
          cl::sycl::range<3>{local, 1, 1}};
      auto aI = bufI.template get_access<cl::sycl::access::mode::read>(h);
      auto aO = bufO.template get_access<cl::sycl::access::mode::write>(h);
      h.parallel_for<typename ExecutionPolicy::kernelName>(
          r, [aI, aO, op, vectorSize](cl::sycl::nd_item<3> id) {
            if ((id.get_global_id(0) < vectorSize)) {
              aO[id.get_global_id(0)] = op(aI[id.get_global_id(0)]);
            }
          });
    };
    q.submit(f);
  }
  return out;
}

/** transform sycl implementation
* @brief Function that takes a Binary Operator and applies to the given range
* @param sep    : Execution Policy
* @param first1 : Start of the range of buffer 1
* @param last1  : End of the range of buffer 1
* @param first2 : Start of the range of buffer 2
* @param result : Output iterator
* @param op     : Binary Operator
* @return  An iterator pointing to the last element
*/
template <class ExecutionPolicy, class InputIterator, class OutputIterator,
          class BinaryOperation>
OutputIterator transform(ExecutionPolicy &sep, InputIterator first1,
                         InputIterator last1, InputIterator first2,
                         OutputIterator result, BinaryOperation op) {
  cl::sycl::queue q(sep.get_queue());
  typedef typename std::iterator_traits<InputIterator>::value_type type_;
  auto buf1 = sycl::helpers::make_const_buffer(first1, last1);
  auto n = buf1.get_count();
  auto buf2 = sycl::helpers::make_const_buffer(first2, first2 + n);
  auto res = sycl::helpers::make_buffer(result, result + n);
  auto f = [n, &buf1, &buf2, &res, op](cl::sycl::handler &h) mutable {
    const size_t local = 128;
    cl::sycl::nd_range<3> r{cl::sycl::range<3>{std::max(n, local), 1, 1},
                            cl::sycl::range<3>{local, 1, 1}};
    auto a1 = buf1.template get_access<cl::sycl::access::mode::read>(h);
    auto a2 = buf2.template get_access<cl::sycl::access::mode::read>(h);
    auto aO = res.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for<typename ExecutionPolicy::kernelName>(
        r, [a1, a2, aO, op, n](cl::sycl::nd_item<3> id) {
          if (id.get_global_id(0) < n) {
            aO[id.get_global_id(0)] =
                op(a1[id.get_global_id(0)], a2[id.get_global_id(0)]);
          }
        });
  };
  q.submit(f);
  return first2 + n;
}

}  // namespace impl
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_TRANSFORM__
