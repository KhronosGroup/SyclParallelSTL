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

#ifndef __SYCL_IMPL_ALGORITHM_MISMATCH__
#define __SYCL_IMPL_ALGORITHM_MISMATCH__

#include <algorithm>
#include <type_traits>

// SYCL helpers header
#include <sycl/algorithm/algorithm_composite_patterns.hpp>
#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_differences.hpp>
#include <sycl/helpers/sycl_namegen.hpp>

namespace sycl {
namespace impl {

/* transform_reduce.
* @brief Returns the transform_reduce of one vector across the range [first1,
* last1) by applying Functions op1 and op2. Implementation of the command
* group
* that submits a transform_reduce kernel.
*/

#ifdef SYCL_PSTL_USE_OLD_ALGO

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
          class BinaryPredicate>
std::pair<ForwardIt1, ForwardIt2> mismatch(ExecutionPolicy& exec,
                                           ForwardIt1 first1, ForwardIt1 last1,
                                           ForwardIt2 first2, ForwardIt2 last2,
                                           BinaryPredicate p) {
  cl::sycl::queue q(exec.get_queue());
  auto size1 = sycl::helpers::distance(first1, last1);
  auto size2 = sycl::helpers::distance(first2, last2);

  cl::sycl::buffer<size_t, 1> bufR((cl::sycl::range<1>(size1)));
  if (size1 < 1 || size2 < 1) {
    return std::make_pair(first1, first2);
  }

  auto device = q.get_device();

  size_t length = std::min(size1, size2);
  auto local = std::min(
      device.template get_info<cl::sycl::info::device::max_work_group_size>(),
      length);
  auto buf1 = sycl::helpers::make_const_buffer(first1, first1 + length);
  auto buf2 = sycl::helpers::make_const_buffer(first2, first2 + length);
  size_t global = exec.calculateGlobalSize(length, local);
  int passes = 0;

  // map across the input testing whether they match the predicate
  auto eqf = [length, local, global, &buf1, &buf2, &bufR,
              p](cl::sycl::handler& h) {
    cl::sycl::nd_range<1> r{cl::sycl::range<1>{std::max(global, local)},
                            cl::sycl::range<1>{local}};
    auto a1 = buf1.template get_access<cl::sycl::access::mode::read>(h);
    auto a2 = buf2.template get_access<cl::sycl::access::mode::read>(h);
    auto aR = bufR.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for<
        cl::sycl::helpers::NameGen<0, typename ExecutionPolicy::kernelName> >(
        r, [a1, a2, aR, length, p](cl::sycl::nd_item<1> id) {
          size_t m_id = id.get_global(0);

          if (m_id < length) aR[m_id] = p(a1[m_id], a2[m_id]) ? length : m_id;
        });
  };
  q.submit(eqf);

  auto binary_op = [](size_t x, size_t y) { return std::min(x, y); };

  do {
    auto f = [passes, length, local, global, &bufR,
              binary_op](cl::sycl::handler& h) mutable {
      cl::sycl::nd_range<1> r{cl::sycl::range<1>{std::max(global, local)},
                              cl::sycl::range<1>{local}};
      auto aR = bufR.template get_access<cl::sycl::access::mode::read_write>(h);
      cl::sycl::accessor<size_t, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          scratch(cl::sycl::range<1>(local), h);

      h.parallel_for<typename ExecutionPolicy::kernelName>(
          r, [aR, scratch, passes, local, length,
              binary_op](cl::sycl::nd_item<1> id) {
            auto r = ReductionStrategy<size_t>(local, length, id, scratch);
            r.workitem_get_from(aR);
            r.combine_threads(binary_op);
            r.workgroup_write_to(aR);
          });
    };
    q.submit(f);
    passes++;
    length = length / local;
  } while (length > 1);
  q.wait_and_throw();
  auto hR = bufR.template get_access<cl::sycl::access::mode::read,
                                     cl::sycl::access::target::host_buffer>();

  auto mismatch_id = hR[0];
  return std::make_pair(first1 + mismatch_id, first2 + mismatch_id);
}

#else

template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
          class BinaryPredicate>
std::pair<ForwardIt1, ForwardIt2> mismatch(ExecutionPolicy& exec,
                                           ForwardIt1 first1, ForwardIt1 last1,
                                           ForwardIt2 first2, ForwardIt2 last2,
                                           BinaryPredicate p) {
  auto size1 = sycl::helpers::distance(first1, last1);
  auto size2 = sycl::helpers::distance(first2, last2);

  if (size1 <= 0 || size2 <= 0) std::make_pair(first1, first2);

  size_t length = std::min(size1, size2);

  auto q = exec.get_queue();

  auto device = q.get_device();
  using value_type1 = typename std::iterator_traits<ForwardIt1>::value_type;
  using value_type2 = typename std::iterator_traits<ForwardIt2>::value_type;

  auto d = compute_mapreduce_descriptor(device, length, sizeof(value_type1));

  auto input_buff1 = sycl::helpers::make_const_buffer(first1, first1 + length);
  auto input_buff2 = sycl::helpers::make_const_buffer(first2, first2 + length);

  auto map = [=](size_t pos, value_type1 x, value_type2 y) {
    return p(x, y) ? length : pos;
  };

  auto red = [](size_t x, size_t y){
    return std::min(x, y);
  };

  auto pos = buffer_map2reduce(exec, q, input_buff1, input_buff2, length, d, map, red);

  return std::make_pair(std::next(first1, pos), std::next(first2, pos));
}

#endif

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_MISMATCH__
