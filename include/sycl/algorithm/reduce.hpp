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
#include <sycl/execution_policy>

namespace sycl {
namespace impl {

/* reduce.
 * Implementation of the command group that submits a reduce kernel.
 * The kernel is implemented as a lambda.
 * Note that there is a potential race condition while using the same buffer for
 * input-output
 */
#ifdef __COMPUTECPP__
template <typename ExecutionPolicy,
          typename Iterator,
          typename T,
          typename BinaryOperation>
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
  return bop(hI[0], init);
}
#else
size_t up_rounded_division(size_t x, size_t y){
  return (x+(y-1)) / y;
}

struct mapreduce_descriptor {
  size_t size, size_per_work_group, nb_work_group, nb_work_item;
  mapreduce_descriptor(size_t size_,
                       size_t size_per_work_group_,
                       size_t nb_work_group_,
                       size_t nb_work_item_):
    size(size_),
    size_per_work_group(size_per_work_group_),
    nb_work_group(nb_work_group_),
    nb_work_item(nb_work_item_) {}
};

/*
 * Compute a valid set of parameters for buffer_mapreduce algorithm to
 * work properly
 */

mapreduce_descriptor compute_mapreduce_descriptor(cl::sycl::device device,
                                                  size_t size,
                                                  size_t sizeofB) {
  using std::max;
  using std::min;
  if (size <= 0)
    return mapreduce_descriptor(0, 0, 0, 0);
  /* Here we have a heuristic which compute appropriate values for the
   * number of work items and work groups, this heuristic ensure that:
   *  - there is less work group than max_compute_units
   *  - there is less work item per work group than max_work_group_size
   * - the memory use to store accumulators of type T is smaller than
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
    return mapreduce_descriptor(size, 0, 0, 0);
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
  size_t size_last_work_group = size % size_per_work_group;
  //std::cout << "size_last_work_group=" << size_last_work_group << std::endl;

  size_t size_per_work_item_last = up_rounded_division(size_last_work_group,
                                                       nb_work_item);

  return mapreduce_descriptor(size, size_per_work_group, nb_work_group,
                              nb_work_item);
}

/*
 * MapReduce Algorithm applied on a buffer
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
                   mapreduce_descriptor d,
                   Map map,
                   Reduce reduce) {

  size_t size = d.size;
  size_t size_per_work_group = d.size_per_work_group;
  size_t nb_work_group = d.nb_work_group;
  size_t nb_work_item = d.nb_work_item;
  /*
   * 'map' is not applied on init
   * 'map': A -> B
   * 'reduce': B * B -> B
   */

  if ((nb_work_item == 0) || (nb_work_group == 0)) {
    auto read_input = input_buff.template get_access
      <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
    B acc = init;
    for (size_t pos = 0; pos < size; pos++)
      acc = reduce(acc, map(pos, read_input[pos]));

    return acc;
    //return boost::compute::reduce(b, e, init, bop);
  }
/*  {
    auto read_input = input_buff.template get_access
      <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
    for(size_t pos = 0; pos < size; pos++)
    {
      std::cout << "[" << pos << "] " << read_input[pos] << ", ";
    }
    std::cout << std::endl;
  }*/

  using std::min;
  using std::max;

  cl::sycl::buffer<B, 1> output_buff { cl::sycl::range<1> { nb_work_group } };

  q.submit([&] (cl::sycl::handler &cgh) {
    cl::sycl::nd_range<1> rng
      { cl::sycl::range<1>{ nb_work_group * nb_work_item },
        cl::sycl::range<1>{ nb_work_item } };
    auto input = input_buff.template get_access
      <cl::sycl::access::mode::read>(cgh);
    auto output = output_buff.template get_access
      <cl::sycl::access::mode::write>(cgh);
    cl::sycl::accessor<B, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
      sum { cl::sycl::range<1>(nb_work_item), cgh };
    cgh.parallel_for_work_group<class wg>(rng, [=](cl::sycl::group<1> grp) {
      //int sum[nb_work_item];
      size_t group_id = grp.get(0);
      assert(group_id < nb_work_group);
      size_t group_begin = group_id * size_per_work_group;
      size_t group_end   = min((group_id+1) * size_per_work_group, size);
      assert(group_begin < group_end); //< as we properly selected the
                                       //  number of work_group
      grp.parallel_for_work_item([&](cl::sycl::nd_item<1> id) {
        size_t local_id = id.get_local(0);
        size_t local_pos = group_begin + local_id;
        if (local_pos < group_end) {
          //we peal the first iteration
          B acc = map(local_pos, input[local_pos]);
          for (size_t read = local_pos + nb_work_item;
               read < group_end;
               read += nb_work_item) {
            acc = reduce(acc, map(read, input[read]));
          }
          sum[local_id] = acc;
        }
      });
      B acc = sum[0];
      for (size_t local_id = 1;
           local_id < min(nb_work_item, group_end - group_begin);
           local_id++)
        acc = reduce(acc, sum[local_id]);

      output[group_id] = acc;
    });
  });
  q.wait_and_throw();
  auto read_output  = output_buff.template get_access
    <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();

  B acc = init;
  for (size_t pos0 = 0; pos0 < nb_work_group; pos0++)
    acc = reduce(acc, read_output[pos0]);

  return acc;
}

/*
 * Map2Reduce on a buffer
 * */


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
                    mapreduce_descriptor d,
                    Map map,
                    Reduce reduce) {
  size_t size = d.size;
  size_t size_per_work_group = d.size_per_work_group;
  size_t nb_work_group = d.nb_work_group;
  size_t nb_work_item = d.nb_work_item;
  /*
   * 'map' is not applied on init
   * 'map': A -> B
   * 'reduce': B * B -> B
   */

  if ((nb_work_item == 0) || (nb_work_group == 0)) {
    auto read_input1 = input_buff1.template get_access
      <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
    auto read_input2 = input_buff2.template get_access
      <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
    B acc = init;
    for (size_t pos = 0; pos < size; pos++)
      acc = reduce(acc, map(pos, read_input1[pos], read_input2[pos]));

    return acc;
    //return boost::compute::reduce(b, e, init, bop);
  }
/*  {
    auto read_input = input_buff.template get_access
      <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
    for(size_t pos = 0; pos < size; pos++)
    {
      std::cout << "[" << pos << "] " << read_input[pos] << ", ";
    }
    std::cout << std::endl;
  }*/

  using std::min;
  using std::max;

  cl::sycl::buffer<B, 1> output_buff { cl::sycl::range<1> { nb_work_group } };

  q.submit([&] (cl::sycl::handler &cgh) {
    cl::sycl::nd_range<1> rng
      { cl::sycl::range<1>{ nb_work_group * nb_work_item },
        cl::sycl::range<1>{ nb_work_item } };
    auto input1  = input_buff1.template get_access
      <cl::sycl::access::mode::read>(cgh);
    auto input2  = input_buff2.template get_access
      <cl::sycl::access::mode::read>(cgh);
    auto output = output_buff.template get_access
      <cl::sycl::access::mode::write>(cgh);
    cl::sycl::accessor<B, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::local>
      sum { cl::sycl::range<1>(nb_work_item), cgh };
    cgh.parallel_for_work_group<class wg>(rng, [=](cl::sycl::group<1> grp) {
      //int sum[nb_work_item];
      size_t group_id = grp.get(0);
      assert(group_id < nb_work_group);
      size_t group_begin = group_id * size_per_work_group;
      size_t group_end = min((group_id+1) * size_per_work_group, size);
      assert(group_begin < group_end); //< as we properly selected the
                                       //  number of work_group
      grp.parallel_for_work_item([&](cl::sycl::nd_item<1> id) {
        size_t local_id = id.get_local(0);
        size_t local_pos = group_begin + local_id;
        if (local_pos < group_end) {
          //we peal the first iteration
          B acc = map(local_pos, input1[local_pos], input2[local_pos]);
          for (size_t read = local_pos + nb_work_item;
               read < group_end;
               read += nb_work_item) {
            acc = reduce(acc, map(read, input1[read], input2[local_pos]));
          }
          sum[local_id] = acc;
        }
      });
      B acc = sum[0];
      for (size_t local_id = 1;
           local_id < min(nb_work_item, group_end - group_begin);
           local_id++)
        acc = reduce(acc, sum[local_id]);

      output[group_id] = acc;
    });
  });
  q.wait_and_throw();
  auto read_output  = output_buff.template get_access
    <cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();

  B acc = init;
  for (size_t pos0 = 0; pos0 < nb_work_group; pos0++)
    acc = reduce(acc, read_output[pos0]);

  return acc;
}

/*
 * Reduce algorithm
 */
template <typename ExecutionPolicy,
          typename Iterator,
          typename T,
          typename BinaryOperation>
typename std::iterator_traits<Iterator>::value_type reduce(
    ExecutionPolicy &snp, Iterator b, Iterator e, T init, BinaryOperation bop) {

  cl::sycl::queue q { snp.get_queue() };
  auto device = q.get_device();
  auto size = sycl::helpers::distance(b, e);
  using value_type = typename std::iterator_traits<Iterator>::value_type;

  if (size <= 0)
    return init;

  mapreduce_descriptor d =
    compute_mapreduce_descriptor(device, size, sizeof(value_type));

  auto input_buff = sycl::helpers::make_const_buffer(b, e);

  auto map = [=](size_t pos, value_type x) { return x; };

  return buffer_mapreduce(snp, q, input_buff, init, d, map, bop);
}

#endif // __COMPUTECPP__

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_REDUCE__
