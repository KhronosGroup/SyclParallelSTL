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

#include "benchmark.h"

using namespace sycl::helpers;

size_t up_rounded_division(size_t x, size_t y) {
  //return x / y + ((x % y == 0) ? 0 : 1);
  return (x+(y-1)) / y;
}

struct mapscan_descriptor {
  size_t size, size_per_work_group, size_per_work_item, nb_work_group,
    nb_work_item;
  mapscan_descriptor(size_t size_, size_t size_per_work_group_,
                     size_t size_per_work_item_, size_t nb_work_group_,
                     size_t nb_work_item_):
    size { size_ },
    size_per_work_group { size_per_work_group_ },
    size_per_work_item { size_per_work_item_ },
    nb_work_group { nb_work_group_ },
    nb_work_item { nb_work_item_ } {}
};

mapscan_descriptor compute_mapscan_descriptor(cl::sycl::device device, size_t size, size_t sizeofB) {
  //std::cout << "size=\t" << size << std::endl; 
  using std::min;
  using std::max;
  if (size == 0) return mapscan_descriptor(0, 0, 0, 0, 0);
  size_t local_mem_size = device.get_info<cl::sycl::info::device::local_mem_size>();
  //std::cout << "local_mem_size=\t" << local_mem_size << std::endl;
  size_t size_per_work_group = min(size, local_mem_size / sizeofB);
  //std::cout << "size_per_work_group=\t" << size_per_work_group << std::endl;
  if (size_per_work_group <= 0) {
    return mapscan_descriptor(size, 0, 0, 0, 0);
  }
  size_t nb_work_group = up_rounded_division(size, size_per_work_group);
  //std::cout << "nb_work_group=\t" << nb_work_group << std::endl;

  size_t max_work_item  = device.get_info<cl::sycl::info::device::max_work_group_size>();
  //std::cout << "max_work_item=\t" << max_work_item << std::endl;
  size_t nb_work_item = min(max_work_item, size_per_work_group);
  //std::cout << "nb_work_item=\t" << nb_work_item << std::endl;
  size_t size_per_work_item = up_rounded_division(size_per_work_group, nb_work_item);
  //std::cout << "size_per_work_item=\t" << size_per_work_item << std::endl;
  return mapscan_descriptor { size, size_per_work_group, size_per_work_item, nb_work_group, nb_work_item };
}


template <typename ExecutionPolicy, typename A, typename B, typename Reduce, class Map>
void buffer_mapscan(
    ExecutionPolicy &snp, cl::sycl::queue q, cl::sycl::buffer<A, 1> input_buffer,
    cl::sycl::buffer<B, 1> output_buffer, B init, mapscan_descriptor d, Map map, Reduce red) {
    //map is not applied on init

  using std::min;
  using std::max;

  size_t size = d.size;
  size_t size_per_work_group = d.size_per_work_group;
  size_t size_per_work_item = d.size_per_work_item;
  size_t nb_work_group = d.nb_work_group;
  size_t nb_work_item = d.nb_work_item;

  //WARNING: nb_work_group is not bounded by max_compute_units
  cl::sycl::buffer<B, 1> scan = { cl::sycl::range<1> { nb_work_group } };

  q.submit([&] (cl::sycl::handler &cgh) {
    cl::sycl::nd_range<1> rng { cl::sycl::range<1>{nb_work_group * nb_work_item},
                                cl::sycl::range<1>{nb_work_item}};
    auto input  = input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
    auto output = output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);
    
    
    // std::cout << "starting STEP I: local scans" << std::endl;

    cgh.parallel_for_work_group<class workgroup>(rng, [=](cl::sycl::group<1> grp) {
      //assert(false);
      B scratch[size_per_work_group];
      size_t group_id = grp.get(0);
      assert(group_id < nb_work_group);
      size_t group_begin = group_id * size_per_work_group;
      size_t group_end   = min((group_id+1) * size_per_work_group, size);
      size_t local_size = group_end - group_begin;
      assert(group_begin < group_end); //as we properly selected the number of work_group

      //step 0:
      //each work_item copy a piece of data
      //map is applied during the process
      grp.parallel_for_work_item([&](cl::sycl::nd_item<1> id) {
        size_t local_id = id.get_local(0);
        //gpos: position in the global vector
        //lpos: position in the local vector
        for (size_t gpos = group_begin + local_id, lpos = local_id;
            gpos < group_end;
            gpos+=nb_work_item, lpos+=nb_work_item) {
          scratch[lpos] = map(input[gpos]);
        }
      });

      //step 1:
      //each work_item scan a piece of data
      grp.parallel_for_work_item([&](cl::sycl::nd_item<1> id) {
        size_t local_id  = id.get_local(0);
        size_t local_pos = local_id * size_per_work_item;
        size_t local_end = min((local_id+1) * size_per_work_item, local_size);
        if (local_pos < local_end) {
          B acc = scratch[local_pos];
          local_pos++;
          for (; local_pos < local_end; local_pos++) {
            scratch[local_pos] = acc = red(acc, scratch[local_pos]);
          }
        }
      });

      //step 2:
      {
        //scan on every last item
        size_t local_pos = size_per_work_item - 1;
        if (local_pos < local_size)
        {
          B acc = scratch[local_pos];
          local_pos += size_per_work_item;
          for (; local_pos < local_size; local_pos += size_per_work_item) {
            scratch[local_pos] = acc = red(acc, scratch[local_pos]);
          }
        }
      }

      //step 3:
      //(except for group = 0) add the last element of the previous block
      grp.parallel_for_work_item([&](cl::sycl::nd_item<1> id) {
        size_t local_id  = id.get_local(0);
        if (local_id > 0) {
          size_t local_pos = local_id * size_per_work_item;
          size_t local_end = min((local_id+1) * size_per_work_item - 1, local_size);
          if (local_pos < local_end) {
            B acc = scratch[local_pos - 1];
            for (; local_pos < local_end; local_pos++) {
              scratch[local_pos] = red(acc, scratch[local_pos]);
            }
          }
        }
      });

      //step 4:
      //each work_item copy a piece of data
      grp.parallel_for_work_item([&](cl::sycl::nd_item<1> id) {
        size_t local_id = id.get_local(0);
        size_t gpos = group_begin + local_id;
        //gpos: position in the global vector
        size_t lpos = local_id;
        //lpos: position in the local vector
        for (size_t gpos = group_begin + local_id, lpos = local_id;
            gpos < group_end;
            gpos+=nb_work_item, lpos+=nb_work_item) {
          output[gpos] = scratch[lpos];
        }
      });

    });
  });
  q.wait_and_throw();

  //STEP II: global scan
  {
    auto buff  = output_buffer.template get_access
      <cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>();
    auto write_scan  = scan.template get_access
      <cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer>();
    //std::cout << "VEC = ["; 
    B acc = init;
    for (size_t global_pos = size_per_work_group - 1, local_pos = 0;
        local_pos < nb_work_group - 1;
        local_pos++, global_pos+=size_per_work_group) {
      write_scan[local_pos] = acc;
      //std::cout << ", " << acc;
      acc = red(acc, buff[global_pos]);
    }
    write_scan[nb_work_group - 1] = acc;
    //std::cout << ", " << acc << "]" << std::endl;
  }


  //STEP III: propagate global scan on local scans
  q.submit([&] (cl::sycl::handler &cgh) {
    cl::sycl::nd_range<1> rng { cl::sycl::range<1>{nb_work_group * nb_work_item},
                                cl::sycl::range<1>{nb_work_item}};
    auto buff  = output_buffer.template get_access<cl::sycl::access::mode::read_write>(cgh);

    //std::cout << "starting STEP III: propagate global scan on local scans" << std::endl;

    auto read_scan  = scan.template get_access<cl::sycl::access::mode::read>(cgh);

    cgh.parallel_for_work_group<class workgroup>(rng, [=](cl::sycl::group<1> grp) {
      size_t group_id = grp.get(0);
      B acc = read_scan[group_id];
      assert(group_id < nb_work_group);
      size_t group_begin = group_id * size_per_work_group;
      size_t group_end   = min((group_id+1) * size_per_work_group, size);
      assert(group_begin < group_end); //as we properly selected the number of work_group

      grp.parallel_for_work_item([&](cl::sycl::nd_item<1> id) {
        size_t local_id = id.get_local(0);
        //gpos: position in the global vector
        //lpos: position in the local vector
        for (size_t gpos = group_begin + local_id;
            gpos < group_end;
            gpos+=nb_work_item) {
          buff[gpos] = red(acc, buff[gpos]);
        }
      });
    });

  });
  //q.wait_and_throw();

}

/** benchmark_inclusive_scan
 * @brief Body Function that executes the SYCL CG of inclusive_scan
 */
benchmark<>::time_units_t benchmark_inclusive_scan(
    const unsigned numReps, const unsigned num_elems,
    const cli_device_selector cds) {
  std::vector<int> vect;

  for (int i = num_elems; i > 0; i--) {
    vect.push_back(i);
  }

  using value_type = int;

  cl::sycl::queue q(cds);
  auto device = q.get_device();
  sycl::sycl_execution_policy<class InclusiveScanAlgorithm1> snp(q);

  auto map = [=](int x) {return x;};
  auto red = [=](value_type x, value_type y) {return x+y;};

  int init = 1;

  std::vector<int> result (vect.size());
  int acc = init;
  for (size_t i = 0; i < vect.size(); i++) {
    result[i] = acc = acc + vect[i];
  }

  auto inclusive_scan = [&]() {

    cl::sycl::buffer<value_type, 1> buffer = { vect.begin(), vect.end() };

    auto d = compute_mapscan_descriptor(device, vect.size(), sizeof(value_type));
    buffer_mapscan(snp, q, buffer, buffer, init, d, map, red);

    auto read_output  = buffer.template get_access
      <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();

    for (size_t i = 0; i < result.size(); i++) {
      assert(result[i] == read_output[i]);
    }
    std::cout << "LOL, it worked !?" << std::endl;

  };

  auto time = benchmark<>::duration(numReps, inclusive_scan);

  return time;
}

BENCHMARK_MAIN("BENCH_SYCL_INCLUSIVE_SCAN", benchmark_inclusive_scan, 2u,
               33554432u, 1);
