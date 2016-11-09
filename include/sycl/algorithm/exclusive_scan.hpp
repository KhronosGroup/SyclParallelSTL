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

#ifndef __SYCL_IMPL_ALGORITHM_EXCLUSIVE_SCAN__
#define __SYCL_IMPL_ALGORITHM_EXCLUSIVE_SCAN__

#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_namegen.hpp>

namespace sycl {
namespace impl {

/* exclusive_scan.
 * Implementation of the command group that submits a exclusive_scan kernel.
 * The kernel is implemented as a lambda.
 */
template <class ExecutionPolicy, class InputIterator, class OutputIterator,
          class ElemT, class BinaryOperation>
OutputIterator exclusive_scan(ExecutionPolicy &sep, InputIterator b,
                              InputIterator e, OutputIterator o, ElemT init,
                              BinaryOperation bop) {
  cl::sycl::queue q(sep.get_queue());
  auto device = q.get_device();

  auto bufI = sycl::helpers::make_const_buffer(b, e);

  auto vectorSize = bufI.get_count();
  // declare a temporary "swap" buffer
  auto bufO = sycl::helpers::make_buffer(o, o + vectorSize);

  size_t localRange =
      std::min(device.get_info<cl::sycl::info::device::max_work_group_size>(),
               vectorSize);
  size_t globalRange = sep.calculateGlobalSize(vectorSize, localRange);
  // calculate iteration count, with extra if not a power of two size buffer
  int iterations = 0;
  for (size_t vs = vectorSize >> 1; vs > 0; vs >>= 1) {
    iterations++;
  }
  if ((vectorSize & (vectorSize - 1)) != 0) {
    iterations++;
  }
  // calculate the buffer to read from first, so we always finally write to bufO
  auto inBuf = &bufI;
  auto outBuf = &bufO;
  if (iterations % 2 != 0) {
    outBuf = &bufI;
    inBuf = &bufO;
  }
  // do a parallel shift right, and set the first element to the initial value.
  // this works, as an exclusive scan is equivalent to a shift
  // (with initial set at element 0) followed by an inclusive scan
  auto shr = [vectorSize, localRange, globalRange, inBuf, outBuf, init](
      cl::sycl::handler &h) {
    cl::sycl::nd_range<3> r{
        cl::sycl::range<3>{std::max(globalRange, localRange), 1, 1},
        cl::sycl::range<3>{localRange, 1, 1}};
    auto aI = inBuf->template get_access<cl::sycl::access::mode::read>(h);
    auto aO = outBuf->template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for<
        cl::sycl::helpers::NameGen<0, typename ExecutionPolicy::kernelName> >(
        r, [aI, aO, init, vectorSize](cl::sycl::nd_item<3> id) {
          int m_id = id.get_global(0);
          if (m_id > 0) {
            aO[m_id] = aI[m_id - 1];
          } else {
            aO[m_id] = init;
          }
        });
  };
  q.submit(shr);
  // swap the buffers so we read from the buffer with the shifted contents
  std::swap(inBuf, outBuf);
  // perform an inclusive scan on the shifted array
  int i = 1;
  do {
    auto f = [vectorSize, i, localRange, globalRange, inBuf, outBuf, bop](
        cl::sycl::handler &h) {
      cl::sycl::nd_range<3> r{
          cl::sycl::range<3>{std::max(globalRange, localRange), 1, 1},
          cl::sycl::range<3>{localRange, 1, 1}};
      auto aI =
          inBuf->template get_access<cl::sycl::access::mode::read_write>(h);
      auto aO =
          outBuf->template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<
          cl::sycl::helpers::NameGen<1, typename ExecutionPolicy::kernelName> >(
          r, [aI, aO, bop, vectorSize, i](cl::sycl::nd_item<3> id) {
            size_t td = 1 << (i - 1);
            size_t m_id = id.get_global(0);
            if (m_id < vectorSize && m_id >= td) {
              aO[m_id] = bop(aI[m_id - td], aI[m_id]);
            } else {
              aO[m_id] = aI[m_id];
            }
          });
    };
    q.submit(f);
    // swap the buffers between iterations
    std::swap(inBuf, outBuf);
    i++;
  } while (i <= iterations);
  q.wait_and_throw();
  return o + vectorSize;
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_EXCLUSIVE_SCAN__
