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

#ifndef __SYCL_IMPL_ALGORITHM_FIND__
#define __SYCL_IMPL_ALGORITHM_FIND__

#include <type_traits>
#include <algorithm>
#include <iostream>
#include <iterator>

// SYCL helpers header
#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/helpers/sycl_namegen.hpp>
#include <sycl/algorithm/algorithm_composite_patterns.hpp>

namespace sycl {
namespace impl {

// a struct to store the result of a predicate comparison, and an index
// we have to declare this here instead of in the function so that the sycl
// kernel can see it
typedef struct search_result {
  bool result;
  int index;
  search_result() = default;
  search_result(bool r, int i) : result(r), index(i) {}
} search_result;

// Implementation of a generic find algorithn to be used for implementing
// the various interfaces specified by the stl
template <class ExecutionPolicy, class InputIt, class UnaryPredicate>
InputIt find_impl(ExecutionPolicy &sep, InputIt b, InputIt e,
                  UnaryPredicate p) {
  cl::sycl::queue q(sep.get_queue());

  auto device = q.get_device();

  // make a buffer that doesn't trigger a copy back, as we don't modify it
  auto buf = sycl::helpers::make_const_buffer(b, e);

  auto vectorSize = buf.get_count();

  // construct a buffer to store the result of the predicate mapping stage
  auto t_buf = sycl::helpers::make_temp_buffer<search_result>(vectorSize);

  if (vectorSize < 1) {
    return e;
  }

  size_t localRange =
      std::min(device.get_info<cl::sycl::info::device::max_work_group_size>(),
               vectorSize);
  size_t globalRange = sep.calculateGlobalSize(vectorSize, localRange);

  // map across the input testing whether they match the predicate
  // store the result of the predicate and the index in the array of the result
  auto eqf = [vectorSize, localRange, globalRange, &buf, &t_buf, p](
      cl::sycl::handler &h) {
    cl::sycl::nd_range<1> r{
        cl::sycl::range<1>{std::max(globalRange, localRange)},
        cl::sycl::range<1>{localRange}};
    auto aI = buf.template get_access<cl::sycl::access::mode::read>(h);
    auto aO = t_buf.template get_access<cl::sycl::access::mode::write>(h);
    h.parallel_for<
        cl::sycl::helpers::NameGen<0, typename ExecutionPolicy::kernelName> >(
        r, [aI, aO, vectorSize, p](cl::sycl::nd_item<1> id) {
          size_t m_id = id.get_global(0);
          // build a pair of equality and index, so that we can find the
          // _first_ index which is true, as opposed to just "one" of them
          aO[m_id] = search_result(p(aI[m_id]), m_id);
        });
  };
  q.submit(eqf);

  // Perform a reduction across the pairs of (predicate result, index), to find
  // the pair with the lowest index where predicate result is true
  // manually implemented, as we can't call currently use sycl buffers with
  // the parallel stl algorithms directly

  // Build a lambda for the reduction, comparing pairs of (result, index) pairs
  // we wish to return the minimum of the two indices where the predicate
  // is "true", or the maximum of the indicies if it is true at neither.
  auto bop = [=](search_result val1, search_result val2) {
    if (val1.result && val2.result) {
      return search_result(true, std::min(val1.index, val2.index));
    } else if (val1.result && !val2.result) {
      return val1;
    } else if (!val1.result && val2.result) {
      return val2;
    } else {
      return search_result(false, std::max(val1.index, val2.index));
    }
  };

  // more or less copied from the reduction implementation
  // TODO: refactor out the implementation details from both into a separate
  // module
  size_t length = vectorSize;
  do {
    auto rf = [length, localRange, globalRange, &t_buf, bop](
        cl::sycl::handler &h) {
      cl::sycl::nd_range<1> r{
          cl::sycl::range<1>{std::max(globalRange, localRange)},
          cl::sycl::range<1>{localRange}};
      auto aI =
          t_buf.template get_access<cl::sycl::access::mode::read_write>(h);
      cl::sycl::accessor<search_result, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>
          scratch(cl::sycl::range<1>(localRange), h);

      h.parallel_for<
          cl::sycl::helpers::NameGen<1, typename ExecutionPolicy::kernelName> >(
          r, [aI, scratch, localRange, length, bop](cl::sycl::nd_item<1> id) {
            auto r = ReductionStrategy<search_result>(localRange, length, id,
                                                      scratch);
            r.workitem_get_from(aI);
            r.combine_threads(bop);
            r.workgroup_write_to(aI);
          });
    };
    q.submit(rf);
    length = length / localRange;
  } while (length > 1);
  q.wait_and_throw();

  auto hI = t_buf.template get_access<cl::sycl::access::mode::read,
                                      cl::sycl::access::target::host_buffer>();

  // there's probably a cleaner way to do this, but essentially once we have
  // the "search index", we need to increment the begin iterator until
  // it reaches that point - we use std::advance, as not all iterators support +
  int search_index = hI[0].index;
  auto r_iter = b;
  std::advance(r_iter, search_index);
  return r_iter;
}
}
}

#endif
