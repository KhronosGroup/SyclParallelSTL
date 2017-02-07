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

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

#include <experimental/algorithm>
#include <sycl/execution_policy>
#include <sycl/algorithm/inclusive_scan.hpp>

#include "benchmark.h"

using namespace sycl::helpers;

/** benchmark_inclusive_scan
 * @brief Body Function that executes the SYCL CG of inclusive_scan
 */
benchmark<>::time_units_t benchmark_inclusive_scan(
    const unsigned numReps, const unsigned num_elems,
    const cli_device_selector cds) {
  std::vector<int> vect (num_elems);
  std::iota(vect.begin(), vect.end(), 0);

  using value_type = int;

  cl::sycl::queue q(cds);
  auto device = q.get_device();
  sycl::sycl_execution_policy<class InclusiveScanAlgorithm1> snp(q);

  int init = 1;

  int result = init;
  for (size_t i = 0; i < vect.size(); i++) {
    result += vect[i];
  }

  auto fun = [=]() {
    assert(result == sycl::impl::accumulate(snp, vect.begin(), vect.end(),
          vect.begin()));
    for (auto i = 0; i < num_elems; ++i) {
      assert(result[i] == vect[i]);
    }
  };

  auto mytime = benchmark<>::duration(numReps, fun);

  return time;
}

BENCHMARK_MAIN("BENCH_SYCL_INCLUSIVE_SCAN", benchmark_inclusive_scan, 2u,
               33554432u, 1);
