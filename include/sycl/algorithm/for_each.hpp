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

#ifndef __SYCL_IMPL_ALGORITHM_FOR_EACH__
#define __SYCL_IMPL_ALGORITHM_FOR_EACH__

#include <type_traits>
#include <algorithm>
#include <iostream>

// SYCL helpers header
#include <sycl/helpers/sycl_buffers.hpp>

namespace sycl {
namespace impl {

/* for_each.
 * Implementation of the command group that submits a for_each kernel.
 * The kernel is implemented as a lambda.
 */
template <class ExecutionPolicy, class Iterator, class UnaryFunction>
void for_each(ExecutionPolicy &sep, Iterator b, Iterator e, UnaryFunction op) {
  {
    cl::sycl::queue q(sep.get_queue());
    auto device = q.get_device();
    size_t localRange =
        device.get_info<cl::sycl::info::device::max_work_group_size>();
    typedef typename std::iterator_traits<Iterator>::value_type type_;
    auto bufI = sycl::helpers::make_buffer(b, e);
    auto vectorSize = bufI.get_count();
    size_t globalRange = sep.calculateGlobalSize(vectorSize, localRange);
    auto f = [vectorSize, localRange, globalRange, &bufI, op](
        cl::sycl::handler &h) mutable {
      cl::sycl::nd_range<3> r{
          cl::sycl::range<3>{std::max(globalRange, localRange), 1, 1},
          cl::sycl::range<3>{localRange, 1, 1}};
      auto aI = bufI.template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<typename ExecutionPolicy::kernelName>(
          r, [aI, op, vectorSize](cl::sycl::nd_item<3> id) {
            if (id.get_global(0) < vectorSize) {
              op(aI[id.get_global(0)]);
            }
          });
    };
    q.submit(f);
  }
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_FOR_EACH__
