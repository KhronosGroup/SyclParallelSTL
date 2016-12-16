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
#include <fstream>
#include <numeric>

#include <experimental/algorithm>
#include <sycl/execution_policy>

#include "benchmark.h"

using namespace sycl::helpers;

size_t up_rounded_division(size_t x, size_t y){
  //return x / y + ((x % y == 0) ? 0 : 1);
  return (x+(y-1)) / y;
}

/** benchmark_reduce
 * @brief Body Function that executes the SYCL CG of Reduce
 */
benchmark<>::time_units_t benchmark_reduce( const unsigned numReps,
                                          const unsigned N,
                                          const cli_device_selector cds) {
  std::vector<int> vect(N);
  std::cout << "vect [" << N << "]" << std::endl;
  //= { ";
  for (int i = 0; i < N; i++) {
    vect[i] = 10 * (((float)std::rand()) / RAND_MAX);
    //std::cout << vect[i] << ", ";
  }
  //std::cout << "}" << std::endl;

  cl::sycl::queue q(cds);

  auto device = q.get_device();
  
  size_t size = N;
  using T = int;
  T init = {};
  auto bop = [=](T x, T y) {return x+y;};

  // We set the number of workgroup to the number of compute units
  
  auto accumulate = [&]() {
    using std::min;
    using std::max;
    if(size == 0) return init;
    /* Here we have a heuristic which compute appropriate values for the number of
     * work items and work groups, this heuristic ensure that:
     *  - there is less work group than max_compute_units
     *  - there is less work item per work group than max_work_group_size
     *  - the memory use to store accumulators of type T is smaller than local_mem_size
     *  - every work group do something
     */
    size_t max_work_group = device.get_info<cl::sycl::info::device::max_compute_units>();
    std::cout << "max_work_group=\t" << max_work_group << std::endl;
    //maximal number of work item per work group
    size_t max_work_item  = device.get_info<cl::sycl::info::device::max_work_group_size>();
    std::cout << "max_work_item=\t" << max_work_item << std::endl;
    size_t local_mem_size = device.get_info<cl::sycl::info::device::local_mem_size>();
    std::cout << "local_mem_size=\t" << local_mem_size << std::endl;
    size_t nb_work_item   = min(max_work_item, local_mem_size / sizeof(T));
    std::cout << "nb_work_item=\t" << nb_work_item << std::endl;

    /* (nb_work_item == 0) iff (sizeof(T) > local_mem_size)
     * If sizeof(T) > local_mem_size, this means that an object
     * of type T can't hold in the memory of a single work-group
     */
    if (nb_work_item == 0) {
      T acc = init;
      for(size_t i = 0; i < size; i++) {
        acc = bop(acc, vect[i]);
      }
      return acc;
      //return std::reduce(vect.begin(), vect.end(), init, bop);
    }
    // we ensure that each work_item of every work_group is used at least once
    size_t nb_work_group = min(max_work_group, up_rounded_division(size, nb_work_item));
    std::cout << "nb_work_group=\t" << nb_work_group << std::endl;
    assert(nb_work_group >= 1);

    //number of elements manipulated by each work_item
    size_t size_per_work_item  = up_rounded_division(size, nb_work_item * nb_work_group);
    std::cout << "size_per_work_item=\t" << size_per_work_item << std::endl;
    //number of elements manipulated by each work_group (except the last one)
    size_t size_per_work_group = size_per_work_item * nb_work_item;
    std::cout << "size_per_work_group=\t" << size_per_work_group << std::endl;

    nb_work_group = max(static_cast<size_t>(1), up_rounded_division(size, size_per_work_group));
    std::cout << "nb_work_group=\t" << nb_work_group << " (updated)" << std::endl;
    assert(nb_work_group >= 1);

    assert(size_per_work_group * (nb_work_group - 1) < size);
    assert(size_per_work_group * nb_work_group >= size);
    /* number of elements manipulated by the last work_group
     * n.b. if the value is 0, the last work_group is regular
     */
    size_t size_last_work_group = size % size_per_work_group;
    std::cout << "size_last_work_group=" << size_last_work_group << std::endl;

    size_t size_per_work_item_last = up_rounded_division(size_last_work_group, nb_work_item);


    sycl::sycl_execution_policy<class ReduceAlgorithmBench> snp(q);
    cl::sycl::buffer<T, 1> input_buff = { vect.cbegin(), vect.cend() };
    cl::sycl::buffer<T, 1> output_buff = { cl::sycl::range<1> { nb_work_group } };

    q.submit([&] (cl::sycl::handler &cgh) {
      cl::sycl::nd_range<1> rng { cl::sycl::range<1>{nb_work_group*nb_work_item},
                                  cl::sycl::range<1>{nb_work_item}};
      auto input  = input_buff.get_access<cl::sycl::access::mode::read>(cgh);
      auto output = output_buff.get_access<cl::sycl::access::mode::write>(cgh);
      cgh.parallel_for_work_group<class workgroup>(rng, [=](cl::sycl::group<1> grp) {
        int sum[nb_work_item];
        size_t group_id = grp.get(0);
        assert(group_id < nb_work_group);
        size_t group_begin = group_id * size_per_work_group;
        size_t group_end   = min((group_id+1) * size_per_work_group, size);
        assert(group_begin < group_end); //as we properly selected the number of work_group
        grp.parallel_for_work_item([&](cl::sycl::nd_item<1> id) {
          size_t local_id = id.get_local(0);
          size_t local_pos = group_begin + local_id;
          if (local_pos < group_end) {
            //we peal the first iteration
            T acc = input[local_pos];
            for(size_t read = local_pos + nb_work_item; read < group_end; read += nb_work_item) {
              acc = bop(acc, input[read]);
            }
            sum[local_id] = acc;
          }
        });
        T acc = sum[0];
        for(size_t local_id = 1; local_id < min(nb_work_item, group_end - group_begin); local_id++) {
          acc = bop(acc, sum[local_id]);
        }
        output[group_id] = acc;
      });
    });
    q.wait_and_throw();
    auto read_output  = output_buff.template get_access
      <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
    
    int acc = 0;
    for(size_t pos0 = 0; pos0 < nb_work_group; pos0++)
    {
      acc = bop(acc, read_output[pos0]);
    }
    return acc;

  };
  auto mainLoop = [&]() {
    auto res = accumulate();
    //auto resstd = std::accumulate(vect.begin(), vect.end(), 0);
    std::cout << "SYCL Result of Reduction is: " << res << std::endl;
    //std::cout << "STL Result of Reduction is: " << resstd << std::endl;
    //assert(res == resstd);
  };
  auto time = benchmark<>::duration(numReps, mainLoop);
  
  auto resstd = std::accumulate(vect.begin(), vect.end(), 0);
  std::cout << "STL Result of Reduction is: " << resstd << std::endl;
  return time;
}

BENCHMARK_MAIN("BENCH_REDUCE", benchmark_reduce, 2, 65536, 1);
