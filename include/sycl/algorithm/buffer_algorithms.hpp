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

#ifndef __SYCL_IMPL_BUFFER_ALGORITHM__
#define __SYCL_IMPL_BUFFER_ALGORITHM__

#include <sycl/helpers/sycl_buffers.hpp>

namespace sycl {
namespace impl {


size_t up_rounded_division(size_t x, size_t y){
  return (x+(y-1)) / y;
}


struct sycl_algorithm_descriptor {
  size_t size,
         size_per_work_group,
         size_per_work_item,
         nb_work_group,
         nb_work_item;
  sycl_algorithm_descriptor() = default;
  sycl_algorithm_descriptor(size_t size_):
    size(size_) {}
  sycl_algorithm_descriptor(size_t size_,
                       size_t size_per_work_group_,
                       size_t size_per_work_item_,
                       size_t nb_work_group_,
                       size_t nb_work_item_):
    size(size_),
    size_per_work_group(size_per_work_group_),
    size_per_work_item(size_per_work_item_),
    nb_work_group(nb_work_group_),
    nb_work_item(nb_work_item_) {}
};

/*
 * Compute a valid set of parameters for buffer_mapreduce algorithm to
 * work properly
 */

sycl_algorithm_descriptor compute_mapreduce_descriptor(cl::sycl::device device,
                                                  size_t size,
                                                  size_t sizeofB) {
  using std::max;
  using std::min;
  if (size <= 0)
    return sycl_algorithm_descriptor {};
  /* Here we have a heuristic which compute appropriate values for the
   * number of work items and work groups, this heuristic ensure that:
   *  - there is less work group than max_compute_units
   *  - there is less work item per work group than max_work_group_size
   *  - the memory use to store accumulators of type T is smaller than
       local_mem_size
   *  - every work group do something
   */
  size_t max_work_group =
    device.get_info<cl::sycl::info::device::max_compute_units>();
  //std::cout << "max_work_group=\t" << max_work_group << std::endl;
  //maximal number of work item per work group
  size_t max_work_item =
    device.get_info<cl::sycl::info::device::max_work_group_size>();
  //std::cout << "max_work_item=\t" << max_work_item << std::endl;
  size_t local_mem_size =
    device.get_info<cl::sycl::info::device::local_mem_size>();
  //std::cout << "local_mem_size=\t" << local_mem_size << std::endl;
  size_t nb_work_item = min(max_work_item, local_mem_size / sizeofB);
  //std::cout << "nb_work_item=\t" << nb_work_item << std::endl;

  /* (nb_work_item == 0) iff (sizeof(T) > local_mem_size)
   * If sizeof(T) > local_mem_size, this means that an object
   * of type T can't hold in the memory of a single work-group
   */
  if (nb_work_item == 0) {
    return sycl_algorithm_descriptor { size };
  }
  // we ensure that each work_item of every work_group is used at least once
  size_t nb_work_group = min(max_work_group,
                             up_rounded_division(size, nb_work_item));
  //std::cout << "nb_work_group=\t" << nb_work_group << std::endl;
  assert(nb_work_group >= 1);

  //number of elements manipulated by each work_item
  size_t size_per_work_item =
    up_rounded_division(size, nb_work_item * nb_work_group);
  //std::cout << "size_per_work_item=\t" << size_per_work_item << std::endl;
  //number of elements manipulated by each work_group (except the last one)
  size_t size_per_work_group = size_per_work_item * nb_work_item;
  //std::cout << "size_per_work_group=\t" << size_per_work_group << std::endl;

  nb_work_group = max(static_cast<size_t>(1),
                      up_rounded_division(size, size_per_work_group));
  //std::cout << "nb_work_group=\t" << nb_work_group
  //          << " (updated)" << std::endl;
  assert(nb_work_group >= 1);

  assert(size_per_work_group * (nb_work_group - 1) < size);
  assert(size_per_work_group * nb_work_group >= size);
  /* number of elements manipulated by the last work_group
   * n.b. if the value is 0, the last work_group is regular
   */
  /*size_t size_last_work_group = size % size_per_work_group;*/
  //std::cout << "size_last_work_group=" << size_last_work_group << std::endl;

  /*size_t size_per_work_item_last = up_rounded_division(size_last_work_group,
                                                       nb_work_item);*/

  return sycl_algorithm_descriptor {
    size,
    size_per_work_group,
    size_per_work_item,
    nb_work_group,
    nb_work_item };
}

/*
 * MapReduce Algorithm applied on a buffer
 *
 * with map/reduce functions typed as follow
 * Map    : A -> B
 * Reduce : B -> B -> B
 */

template <typename ExecutionPolicy,
          typename A,
          typename B,
          typename Reduce,
          typename Map>
B buffer_mapreduce(ExecutionPolicy &snp,
                   cl::sycl::queue q,
                   cl::sycl::buffer<A, 1> input_buff,
                   B init, //map is not applied on init
                   sycl_algorithm_descriptor d,
                   Map map,
                   Reduce reduce) {

  /*
   * 'map' is not applied on init
   * 'map': A -> B
   * 'reduce': B * B -> B
   */

  if ((d.nb_work_item == 0) || (d.nb_work_group == 0)) {
    auto read_input = input_buff.template get_access
      <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
    B acc = init;
    for (size_t pos = 0; pos < d.size; pos++)
      acc = reduce(acc, map(pos, read_input[pos]));

    return acc;
  }

  using std::min;
  using std::max;

  cl::sycl::buffer<B, 1> output_buff { cl::sycl::range<1> { d.nb_work_group } };

  q.submit([&] (cl::sycl::handler &cgh) {
    cl::sycl::range<1> rg { d.nb_work_group * d.nb_work_item };
    cl::sycl::range<1> ri { d.nb_work_item };
    auto input = input_buff.template get_access
      <cl::sycl::access::mode::read>(cgh);
    auto output = output_buff.template get_access
      <cl::sycl::access::mode::write>(cgh);
    cl::sycl::accessor<B, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
      sum { cl::sycl::range<1>(d.nb_work_item), cgh };
    cgh.parallel_for_work_group<class wg>(rg, ri, [=](cl::sycl::group<1> grp) {
      size_t group_id = grp.get(0);
      //assert(group_id < d.nb_work_group);
      size_t group_begin = group_id * d.size_per_work_group;
      size_t group_end   = min((group_id+1) * d.size_per_work_group, d.size);
      //assert(group_begin < group_end); //< as we properly selected the
                                       //  number of work_group
      parallel_for_work_item(grp, [&](cl::sycl::item<1> id) {
        size_t local_id = id.get(0) % d.nb_work_item;
        size_t local_pos = group_begin + local_id;
        if (local_pos < group_end) {
          //we peal the first iteration
          B acc = map(local_pos, input[local_pos]);
          for (size_t read = local_pos + d.nb_work_item;
               read < group_end;
               read += d.nb_work_item) {
            acc = reduce(acc, map(read, input[read]));
          }
          sum[local_id] = acc;
        }
      });
      B acc = sum[0];
      for (size_t local_id = 1;
           local_id < min(d.nb_work_item, group_end - group_begin);
           local_id++)
        acc = reduce(acc, sum[local_id]);

      output[group_id] = acc;
    });
  });
  auto read_output  = output_buff.template get_access
    <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();

  B acc = init;
  for (size_t pos0 = 0; pos0 < d.nb_work_group; pos0++)
    acc = reduce(acc, read_output[pos0]);

  return acc;
}

/*
 * Map2Reduce on a buffer
 *
 * with mat/reduce typed as follow:
 * Map : A1 -> A2 -> B
 * Reduce : B -> B -> B
 *
 */


template <typename ExecutionPolicy,
          typename A1,
          typename A2,
          typename B,
          typename Reduce,
          typename Map>
B buffer_map2reduce(ExecutionPolicy &snp,
                    cl::sycl::queue q,
                    cl::sycl::buffer<A1, 1> input_buff1,
                    cl::sycl::buffer<A2, 1> input_buff2,
                    B init, //map is not applied on init
                    sycl_algorithm_descriptor d,
                    Map map,
                    Reduce reduce) {

  if ((d.nb_work_item == 0) || (d.nb_work_group == 0)) {
    auto read_input1 = input_buff1.template get_access
      <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
    auto read_input2 = input_buff2.template get_access
      <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
    B acc = init;
    for (size_t pos = 0; pos < d.size; pos++)
      acc = reduce(acc, map(pos, read_input1[pos], read_input2[pos]));

    return acc;
  }

  using std::min;
  using std::max;

  cl::sycl::buffer<B, 1> output_buff { cl::sycl::range<1> { d.nb_work_group } };

  q.submit([&] (cl::sycl::handler &cgh) {
    cl::sycl::nd_range<1> rng
      { cl::sycl::range<1>{ d.nb_work_group * d.nb_work_item },
        cl::sycl::range<1>{ d.nb_work_item } };
    auto input1  = input_buff1.template get_access
      <cl::sycl::access::mode::read>(cgh);
    auto input2  = input_buff2.template get_access
      <cl::sycl::access::mode::read>(cgh);
    auto output = output_buff.template get_access
      <cl::sycl::access::mode::write>(cgh);
    cl::sycl::accessor<B, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
      sum { cl::sycl::range<1>(d.nb_work_item), cgh };
    cgh.parallel_for_work_group<class wg>(rng, [=](cl::sycl::group<1> grp) {
      size_t group_id = grp.get(0);
      //assert(group_id < d.nb_work_group);
      size_t group_begin = group_id * d.size_per_work_group;
      size_t group_end = min((group_id+1) * d.size_per_work_group, d.size);
      //assert(group_begin < group_end); // as we properly selected the
                                         // number of work_group
      parallel_for_work_item(grp, [&](cl::sycl::item<1> id) {
        size_t local_id = id.get(0) % d.nb_work_item;
        size_t local_pos = group_begin + local_id;
        if (local_pos < group_end) {
          //we peal the first iteration
          B acc = map(local_pos, input1[local_pos], input2[local_pos]);
          for (size_t read = local_pos + d.nb_work_item;
               read < group_end;
               read += d.nb_work_item) {
            acc = reduce(acc, map(read, input1[read], input2[local_pos]));
          }
          sum[local_id] = acc;
        }
      });
      B acc = sum[0];
      for (size_t local_id = 1;
           local_id < min(d.nb_work_item, group_end - group_begin);
           local_id++)
        acc = reduce(acc, sum[local_id]);

      output[group_id] = acc;
    });
  });
  auto read_output  = output_buff.template get_access
    <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();

  B acc = init;
  for (size_t pos0 = 0; pos0 < d.nb_work_group; pos0++)
    acc = reduce(acc, read_output[pos0]);

  return acc;
}



sycl_algorithm_descriptor compute_mapscan_descriptor(cl::sycl::device device,
                                              size_t size,
                                              size_t sizeofB) {
  //std::cout << "size=\t" << size << std::endl;
  using std::min;
  using std::max;
  if (size == 0)
    return sycl_algorithm_descriptor {};
  size_t local_mem_size =
    device.get_info<cl::sycl::info::device::local_mem_size>();
  //std::cout << "local_mem_size=\t" << local_mem_size << std::endl;
  size_t size_per_work_group = min(size, local_mem_size / sizeofB);
  //std::cout << "size_per_work_group=\t" << size_per_work_group << std::endl;
  if (size_per_work_group <= 0)
    return sycl_algorithm_descriptor { size };

  size_t nb_work_group = up_rounded_division(size, size_per_work_group);
  //std::cout << "nb_work_group=\t" << nb_work_group << std::endl;

  size_t max_work_item =
    device.get_info<cl::sycl::info::device::max_work_group_size>();
  //std::cout << "max_work_item=\t" << max_work_item << std::endl;
  size_t nb_work_item = min(max_work_item, size_per_work_group);
  //std::cout << "nb_work_item=\t" << nb_work_item << std::endl;
  size_t size_per_work_item =
    up_rounded_division(size_per_work_group, nb_work_item);
  //std::cout << "size_per_work_item=\t" << size_per_work_item << std::endl;
  return sycl_algorithm_descriptor {
    size,
    size_per_work_group,
    size_per_work_item,
    nb_work_group,
    nb_work_item };
}


template <class ExecutionPolicy, class A, class B, class Reduce, class Map>
void buffer_mapscan(ExecutionPolicy &snp,
                    cl::sycl::queue q,
                    cl::sycl::buffer<A, 1> input_buffer,
                    cl::sycl::buffer<B, 1> output_buffer,
                    B init,
                    sycl_algorithm_descriptor d,
                    Map map,
                    Reduce red) {
    //map is not applied on init

  using std::min;
  using std::max;

  //WARNING: nb_work_group is not bounded by max_compute_units
  auto scan = sycl::helpers::make_temp_buffer<B>( d.nb_work_group );
  //cl::sycl::buffer<B, 1> scan = { cl::sycl::range<1> { d.nb_work_group } };
  cl::sycl::range<1> rng_wg {d.nb_work_group * d.nb_work_item};
  cl::sycl::range<1> rng_wi {d.nb_work_item};

  q.submit([&] (cl::sycl::handler &cgh) {
    auto input =
      input_buffer.template get_access<cl::sycl::access::mode::read>(cgh);
    auto output =
      output_buffer.template get_access<cl::sycl::access::mode::write>(cgh);

    cl::sycl::accessor<B, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
      scratch { cl::sycl::range<1> { d.size_per_work_group }, cgh };

    cgh.parallel_for_work_group<class wg>(rng_wg, rng_wi,
                                          [=](cl::sycl::group<1> grp) {
      size_t group_id = grp.get(0);
      //assert(group_id < d.nb_work_group);
      size_t group_begin = group_id * d.size_per_work_group;
      size_t group_end   = min((group_id+1) * d.size_per_work_group, d.size);
      size_t local_size = group_end - group_begin;
      //assert(group_begin < group_end); //  as we properly selected the
                                         //  number of work_group

      // Step 0:
      // each work_item copy a piece of data
      // map is applied during the process
      parallel_for_work_item(grp, [&](cl::sycl::item<1> id) {
        size_t local_id  = id.get(0) % d.nb_work_item;
        // gpos: position in the global vector
        // lpos: position in the local vector
        for (size_t gpos = group_begin + local_id, lpos = local_id;
            gpos < group_end;
            gpos += d.nb_work_item, lpos += d.nb_work_item) {
          scratch[lpos] = map(input[gpos]);
        }
      });

      // Step 1:
      // each work_item scan a piece of data
      parallel_for_work_item(grp, [&](cl::sycl::item<1> id) {
        size_t local_id  = id.get(0) % d.nb_work_item;
        size_t local_pos = local_id * d.size_per_work_item;
        size_t local_end = min((local_id+1) * d.size_per_work_item, local_size);
        if (local_pos < local_end) {
          B acc = scratch[local_pos];
          local_pos++;
          for (; local_pos < local_end; local_pos++) {
            acc = red(acc, scratch[local_pos]);
            scratch[local_pos] = acc;
          }
        }
      });

      // Step 2:
      {
        // scan on every last item
        size_t local_pos = d.size_per_work_item - 1;
        if (local_pos < local_size)
        {
          B acc = scratch[local_pos];
          local_pos += d.size_per_work_item;
          for (; local_pos < local_size; local_pos += d.size_per_work_item) {
            acc = red(acc, scratch[local_pos]);
            scratch[local_pos] = acc;
          }
        }
      }

      // Step 3:
      // (except for group = 0) add the last element of the previous block
      parallel_for_work_item(grp, [&](cl::sycl::item<1> id) {
        size_t local_id  = id.get(0) % d.nb_work_item;
        if (local_id > 0) {
          size_t local_pos = local_id * d.size_per_work_item;
          size_t local_end = min((local_id+1) * d.size_per_work_item - 1,
                                 local_size);
          if (local_pos < local_end) {
            B acc = scratch[local_pos - 1];
            for (; local_pos < local_end; local_pos++) {
              scratch[local_pos] = red(acc, scratch[local_pos]);
            }
          }
        }
      });

      // Step 4:
      // each work_item copy a piece of data
      parallel_for_work_item(grp, [&](cl::sycl::item<1> id) {
        size_t local_id = id.get(0) % d.nb_work_item;
        size_t gpos = group_begin + local_id;
        // gpos: position in the global vector
        size_t lpos = local_id;
        // lpos: position in the local vector
        for (size_t gpos = group_begin + local_id, lpos = local_id;
            gpos < group_end;
            gpos+=d.nb_work_item, lpos+=d.nb_work_item) {
          output[gpos] = scratch[lpos];
        }
      });

    });
  });

  // STEP II: global scan
  {
    auto buff  = output_buffer.template get_access
      <cl::sycl::access::mode::read_write,
       cl::sycl::access::target::host_buffer>();
    auto write_scan  = scan.template get_access
      <cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer>();
    B acc = init;
    for (size_t global_pos = d.size_per_work_group - 1, local_pos = 0;
        local_pos < d.nb_work_group - 1;
        local_pos++, global_pos += d.size_per_work_group) {
      write_scan[local_pos] = acc;
      acc = red(acc, buff[global_pos]);
    }
    write_scan[d.nb_work_group - 1] = acc;
  }


  // STEP III: propagate global scan on local scans
  q.submit([&] (cl::sycl::handler &cgh) {
    auto buff = output_buffer.template get_access
      <cl::sycl::access::mode::read_write>(cgh);
    auto read_scan = scan.template get_access
      <cl::sycl::access::mode::read>(cgh);

    cgh.parallel_for_work_group<class wg>(rng_wg, rng_wi,
                                          [=](cl::sycl::group<1> grp) {
      size_t group_id = grp.get(0);
      B acc = read_scan[group_id];
      //assert(group_id < d.nb_work_group);
      size_t group_begin = group_id * d.size_per_work_group;
      size_t group_end   = min((group_id+1) * d.size_per_work_group, d.size);
      //assert(group_begin < group_end); //  as we properly selected the
                                         //  number of work_group

      parallel_for_work_item(grp, [&](cl::sycl::item<1> id) {
        size_t local_id = id.get(0) % d.nb_work_item;
        // gpos: position in the global vector
        // lpos: position in the local vector
        for (size_t gpos = group_begin + local_id;
             gpos < group_end;
             gpos += d.nb_work_item) {
          buff[gpos] = red(acc, buff[gpos]);
        }
      });
    });

  });

  return;
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_BUFFER_ALGORITHM__
