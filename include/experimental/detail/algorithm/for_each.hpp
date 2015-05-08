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
#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_FOR_EACH__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_FOR_EACH__

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

template <class ExecutionPolicy, class Iterator,
          class UnaryFunction>
void for_each(ExecutionPolicy &sep, Iterator b, Iterator e, 
                        UnaryFunction op) {
  {
    cl::sycl::queue q(sep.get_queue());
    typedef typename std::iterator_traits<Iterator>::value_type type_;
    auto bufI = make_buffer(b, e);
    auto vectorSize = bufI.get_count();
    auto f = [vectorSize, &bufI, op](cl::sycl::handler &h) mutable {
      cl::sycl::range<3> r{ vectorSize, 1, 1 };
      auto aI = bufI.template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<typename ExecutionPolicy::kernelName>(r,
          [aI, op](cl::sycl::id<3> id) {
              op(aI[id.get(0)]);
//              aI[id.get(0)] = 3;
            });
    };
    q.submit(f);
  }
}

}  // namespace detail 
}  // namespace sycl
} // namespace parallel
} // namespace experimental
} // namespace std

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_FOR_EACH__
