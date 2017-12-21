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

#ifndef __SYCL_IMPL_ALGORITHM_REPLACE_IF__
#define __SYCL_IMPL_ALGORITHM_REPLACE_IF__

#include <algorithm>
#include <iostream>
#include <type_traits>

// SYCL helpers header
#include <sycl/helpers/sycl_buffers.hpp>

namespace sycl {
namespace impl {

/* replace_if.
 * Implementation of the command group that submits a replace_if kernel.
 * The kernel is implemented as a lambda.
 */
template <class ExecutionPolicy, class ForwardIt, class UnaryPredicate, class T>
void replace_if(ExecutionPolicy &sep, ForwardIt first, ForwardIt last,
                UnaryPredicate p, const T &new_value) {
  cl::sycl::queue q{sep.get_queue()};
  auto device = q.get_device();
  size_t localRange =
      device.get_info<cl::sycl::info::device::max_work_group_size>();
  auto bufI = helpers::make_buffer(first, last);

  // copy new_value, as we cannot capture it by reference
  T new_val = new_value;

  auto vectorSize = bufI.get_count();
  size_t globalRange = sep.calculateGlobalSize(vectorSize, localRange);
  auto f = [vectorSize, p, new_val, localRange, globalRange,
            &bufI](cl::sycl::handler &h) mutable {
    cl::sycl::nd_range<1> r{
        cl::sycl::range<1>{std::max(globalRange, localRange)},
        cl::sycl::range<1>{localRange}};

    auto aI = bufI.template get_access<cl::sycl::access::mode::read_write>(h);
    h.parallel_for<typename ExecutionPolicy::kernelName>(
        r, [aI, vectorSize, p, new_val](cl::sycl::nd_item<1> id) {
          auto global_id = id.get_global(0);
          if (global_id < vectorSize) {
            if(p(aI[global_id])) {
              aI[global_id] = new_val;
            }
          }
        });
  };
  q.submit(f);
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_REPLACE_IF__
