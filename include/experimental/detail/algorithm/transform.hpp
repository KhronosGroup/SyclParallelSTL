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

#include <experimental/execution_policy>
// Detail header
#include <experimental/detail/sycl_buffers.hpp>

namespace std {
namespace experimental {
namespace parallel {
namespace sycl {
namespace detail {

/* transform.
 * Implementation of the command group that submits a transform kernel.
 * The kernel is implemented as a lambda.
 */
template <class ExecutionPolicy, class Iterator, class OutputIterator,
          class UnaryOperation>
OutputIterator transform(ExecutionPolicy &sep, Iterator b, Iterator e,
                         OutputIterator out_b, UnaryOperation op) {
  {
    cl::sycl::queue q(sep.get_queue());
    typedef typename std::iterator_traits<Iterator>::value_type type_;
    auto bufI = make_const_buffer(b, e);
    auto bufO = make_buffer(out_b, out_b + bufI.get_count());
    auto vectorSize = bufI.get_count();
    auto f = [vectorSize, &bufI, &bufO, op](cl::sycl::handler &h) mutable {
      cl::sycl::range<3> r{ vectorSize, 1, 1 };
      auto aI = bufI.template get_access<cl::sycl::access::mode::read>(h);
      auto aO = bufO.template get_access<cl::sycl::access::mode::write>(h);
      h.parallel_for<typename ExecutionPolicy::kernelName>(r,
          [aI, aO, op](cl::sycl::id<3> id) {
              aO[id.get(0)] = op(aI[id.get(0)]);
            });
    };
    q.submit(f);
  }
  return out_b;
}

}  // namespace detail 
}  // namespace sycl
} // namespace parallel
} // namespace experimental
} // namespace std

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_TRANSFORM__
