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

#ifndef __SYCL_IMPL_ALGORITHM_REDUCE__
#define __SYCL_IMPL_ALGORITHM_REDUCE__

#include <type_traits>
#include <typeinfo>
#include <algorithm>
#include <iostream>

// SYCL helpers header
#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_differences.hpp>
#include <sycl/algorithm/algorithm_composite_patterns.hpp>

namespace sycl {
namespace impl {

/* reduce.
 * Implementation of the command group that submits a reduce kernel.
 * The kernel is implemented as a lambda.
 * Note that there is a potential race condition while using the same buffer for
 * input-output
 */
#if CL_SYCL_LANGUAGE_VERSION < 220
template <class ExecutionPolicy, class Iterator, class T, class BinaryOperation>
typename std::iterator_traits<Iterator>::value_type reduce(
    ExecutionPolicy &sep, Iterator b, Iterator e, T init, BinaryOperation bop) {
  cl::sycl::queue q(sep.get_queue());

  auto vectorSize = sycl::helpers::distance(b, e);

  if (vectorSize < 1) {
    return init;
  }

  auto device = q.get_device();
  auto local =
      std::min(device.get_info<cl::sycl::info::device::max_work_group_size>(),
               vectorSize);

  typedef typename std::iterator_traits<Iterator>::value_type type_;
  auto bufI = sycl::helpers::make_const_buffer(b, e);
  size_t length = vectorSize;
  size_t global = sep.calculateGlobalSize(length, local);

  do {
    auto f = [length, local, global, &bufI, bop](cl::sycl::handler &h) mutable {
      cl::sycl::nd_range<1> r{cl::sycl::range<1>{std::max(global, local)},
                              cl::sycl::range<1>{local}};
      auto aI = bufI.template get_access<cl::sycl::access::mode::read_write>(h);
      cl::sycl::accessor<type_, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          scratch(cl::sycl::range<1>(local), h);

      h.parallel_for<typename ExecutionPolicy::kernelName>(
          r, [aI, scratch, local, length, bop](cl::sycl::nd_item<1> id) {
            auto r = ReductionStrategy<T>(local, length, id, scratch);
            r.workitem_get_from(aI);
            r.combine_threads(bop);
            r.workgroup_write_to(aI);
          });
    };
    q.submit(f);
    length = length / local;
  } while (length > 1);
  q.wait_and_throw();
  auto hI = bufI.template get_access<cl::sycl::access::mode::read,
                                     cl::sycl::access::target::host_buffer>();
  return hI[0] + init;
}
#else
template <class ExecutionPolicy, class Iterator, class T, class BinaryOperation>
typename std::iterator_traits<Iterator>::value_type reduce(
    ExecutionPolicy &sep, Iterator b, Iterator e, T init, BinaryOperation bop) {

	typedef typename std::iterator_traits<Iterator>::value_type type_;

	cl::sycl::queue q(sep.get_queue());
	
	auto vectorSize = sycl::helpers::distance(b, e);
	
	if (vectorSize < 1) {
		return init;
	}

	auto device = q.get_device();
	auto nb_work_item   = device.get_info<cl::sycl::info::device::max_work_group_size>();

	auto buff = sycl::helpers::make_const_buffer(b, e);
	
	size_t length = vectorSize;
	while (length > 1) {
		q.submit([&] (cl::sycl::handler &cgh) {
			cl::sycl::nd_range<1> rng { cl::sycl::range<1>{std::max(length, nb_work_item)},
										cl::sycl::range<1>{nb_work_item}};

			auto buff_acc = buff.template get_access<cl::sycl::access::mode::read_write>(cgh);

			cl::sycl::accessor<type_, 1, cl::sycl::access::mode::read_write,
			                             cl::sycl::access::target::local> scratch(cl::sycl::range<1>(nb_work_item), cgh);
			cgh.parallel_for_work_group<typename ExecutionPolicy::kernelName>(rng, [=](cl::sycl::group<1> grp) {
				
				// Step 1:
				//     copy a sub-vector of 'buff' into 'scratch'
				grp.parallel_for_work_item([&](cl::sycl::nd_item<1> id) {
					int globalid = id.get_global(0);
					int localid = id.get_local(0);
					if (globalid < length) {
						scratch[localid] = buff_acc[globalid];
					}
				});	
				
				// Step 2:
				//     reduce 'scratch' with the associative, commutative, binary operator 'bop'
				//     use a parallel divide and conquer strategy
				int min = (length < nb_work_item) ? length : nb_work_item;
				for (int offset = min >> 1; offset > 0; offset = offset >> 1) {
					grp.parallel_for_work_item([&](cl::sycl::nd_item<1> id){
						int localid = id.get_local(0);
						if (localid < offset) {
							scratch[localid] = bop(scratch[localid], scratch[localid + offset]);
						}
					});
				}

				// Step 3:
				//     copy back the reduction of 'scratch' into 'buff'
				buff_acc[grp.get(0)] = scratch[0];

			});
			
		});
		q.wait_and_throw();
		length /= nb_work_item;
	}
	auto mybuff_read = buff.template get_access< cl::sycl::access::mode::read > ();
	return bop(init, mybuff_read[0]);
}

#endif // __COMPUTECPP__

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_REDUCE__
