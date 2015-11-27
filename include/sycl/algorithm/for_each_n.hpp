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

#ifndef __EXPERIMENTAL_DETAIL_ALGORITHM_FOR_EACH_N__
#define __EXPERIMENTAL_DETAIL_ALGORITHM_FOR_EACH_N__

#include <type_traits>
#include <algorithm>
#include <iostream>

namespace sycl {
namespace impl {

/* for_each_n.
* @brief Applies a Function across the range [first, first + n).
* Implementation of the command group that submits a for_each_n kernel,
* According to Parallelism TS version n4507. Section 4.3.2
* The kernel is implemented as a lambda.
* @param ExecutionPolicy exec : The execution policy to be used
* @param InputIterator first : Start of the range via a forward iterator
* @param Size n : Specifies the number of valid elements
* @param Function  f : No restrictions
*/
template <class ExecutionPolicy, class InputIterator, class Size,
          class Function>
InputIterator for_each_n(ExecutionPolicy &exec, InputIterator first, Size n,
                         Function f) {
  cl::sycl::queue q(exec.get_queue());
  if (n > 0) {
    auto last(first + n);
    auto device = q.get_device();
    size_t local =
        device.get_info<cl::sycl::info::device::max_work_group_size>();
    auto bufI = sycl::helpers::make_buffer(first, last);
    auto vectorSize = bufI.get_count();
    size_t global = exec.calculateGlobalSize(vectorSize, local);
    auto cg =
        [vectorSize, local, global, &bufI, f](cl::sycl::handler &h) mutable {
      cl::sycl::nd_range<3> r{cl::sycl::range<3>{std::max(global, local), 1, 1},
                              cl::sycl::range<3>{local, 1, 1}};
      auto aI = bufI.template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<typename ExecutionPolicy::kernelName>(
          r, [vectorSize, aI, f](cl::sycl::nd_item<3> id) {
            if (id.get_global(0) < vectorSize) {
              f(aI[id.get_global(0)]);
            }
          });
    };
    q.submit(cg);
    return last;
  } else {
    return first;
  }
}

}  // end impl
}  // namespace sycl

#endif  // __EXPERIMENTAL_DETAIL_ALGORITHM_FOR_EACH_N__
