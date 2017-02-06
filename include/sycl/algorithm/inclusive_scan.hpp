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

#ifndef __SYCL_IMPL_ALGORITHM_INCLUSIVE_SCAN__
#define __SYCL_IMPL_ALGORITHM_INCLUSIVE_SCAN__

#include <sycl/helpers/sycl_buffers.hpp>
#include <sycl/algorithm/reduce.hpp>

namespace sycl {
namespace impl {

#if 0
/* inclusive_scan.
 * Implementation of the command group that submits a inclusive_scan kernel.
 * The kernel is implemented as a lambda.
 */
 template <class ExecutionPolicy, class InputIterator, class OutputIterator,
          class T, class BinaryOperation>
OutputIterator inclusive_scan(ExecutionPolicy &sep, InputIterator b,
                              InputIterator e, OutputIterator o, T init,
                              BinaryOperation bop) {
  cl::sycl::queue q(sep.get_queue());
  auto device = q.get_device();
  // limits us to random access iterators :/
  *b = bop(*b, init);
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
  // calculate the buffer to read from first, based on modulo arithmetic of the
  // required iteration count so we always finally write to bufO
  // implementation based on "naive" implementation from
  // http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
  auto inBuf = &bufI;
  auto outBuf = &bufO;
  if (iterations % 2 == 0) {
    outBuf = &bufI;
    inBuf = &bufO;
  }

  int i = 1;
  do {
    auto f = [vectorSize, i, localRange, globalRange, inBuf, outBuf, bop](
        cl::sycl::handler &h) {
      cl::sycl::nd_range<1> r{
          cl::sycl::range<1>{std::max(globalRange, localRange)},
          cl::sycl::range<1>{localRange}};
      auto aI =
          inBuf->template get_access<cl::sycl::access::mode::read_write>(h);
      auto aO =
          outBuf->template get_access<cl::sycl::access::mode::read_write>(h);
      h.parallel_for<typename ExecutionPolicy::kernelName>(
          r, [aI, aO, bop, vectorSize, i](cl::sycl::nd_item<1> id) {
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
    // swap the buffers
    std::swap(inBuf, outBuf);
    i++;
  } while (i <= iterations);
  q.wait_and_throw();
  return o + vectorSize;
}
#else


typedef struct mapscan_descriptor {
  size_t size, size_per_work_group, size_per_work_item, nb_work_group, nb_work_item;
  mapscan_descriptor(size_t size_, size_t size_per_work_group_ , size_t size_per_work_item_, size_t nb_work_group_, size_t nb_work_item_):
    size(size_),
    size_per_work_group(size_per_work_group_),
    size_per_work_item(size_per_work_item_),
    nb_work_group(nb_work_group_),
    nb_work_item(nb_work_item_) {}
} mapscan_descriptor;

mapscan_descriptor compute_mapscan_descriptor(cl::sycl::device device, size_t size, size_t sizeofB) {
  //std::cout << "size=\t" << size << std::endl; 
  using std::min;
  using std::max;
  if(size == 0) return mapscan_descriptor(0, 0, 0, 0, 0);
  size_t local_mem_size = device.get_info<cl::sycl::info::device::local_mem_size>();
  //std::cout << "local_mem_size=\t" << local_mem_size << std::endl;
  size_t size_per_work_group = min(size, local_mem_size / sizeofB);
  //std::cout << "size_per_work_group=\t" << size_per_work_group << std::endl;
  if(size_per_work_group <= 0) {
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
  return mapscan_descriptor(size, size_per_work_group, size_per_work_item, nb_work_group, nb_work_item);
}


template <class ExecutionPolicy, class A, class B, class Reduce, class Map>
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
        for(size_t gpos = group_begin + local_id, lpos = local_id;
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
        if(local_pos < local_end) {
          B acc = scratch[local_pos];
          local_pos++;
          for(; local_pos < local_end; local_pos++) {
            scratch[local_pos] = acc = red(acc, scratch[local_pos]);
          }
        }
      });

      //step 2:
      {
        //scan on every last item
        size_t local_pos = size_per_work_item - 1;
        if(local_pos < local_size)
        {
          B acc = scratch[local_pos];
          local_pos += size_per_work_item;
          for(; local_pos < local_size; local_pos += size_per_work_item){
            scratch[local_pos] = acc = red(acc, scratch[local_pos]);
          }
        }
      }

      //step 3:
      //(except for group = 0) add the last element of the previous block
      grp.parallel_for_work_item([&](cl::sycl::nd_item<1> id) {
        size_t local_id  = id.get_local(0);
        if(local_id > 0) {
          size_t local_pos = local_id * size_per_work_item;
          size_t local_end = min((local_id+1) * size_per_work_item - 1, local_size);
          if(local_pos < local_end) {
            B acc = scratch[local_pos - 1];
            for(; local_pos < local_end; local_pos++) {
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
        for(size_t gpos = group_begin + local_id, lpos = local_id;
            gpos < group_end;
            gpos+=nb_work_item, lpos+=nb_work_item) {
          output[gpos] = scratch[lpos];
        }
      });

    });
  });

  //STEP II: global scan
  {
    auto buff  = output_buffer.template get_access
      <cl::sycl::access::mode::read_write, cl::sycl::access::target::host_buffer>();
    auto write_scan  = scan.template get_access
      <cl::sycl::access::mode::write, cl::sycl::access::target::host_buffer>();
    B acc = init;
    for(size_t global_pos = size_per_work_group - 1, local_pos = 0;
        local_pos < nb_work_group - 1;
        local_pos++, global_pos+=size_per_work_group){
      write_scan[local_pos] = acc;
      acc = red(acc, buff[global_pos]);
    }
    write_scan[nb_work_group - 1] = acc;
  }


  //STEP III: propagate global scan on local scans
  q.submit([&] (cl::sycl::handler &cgh) {
    cl::sycl::nd_range<1> rng { cl::sycl::range<1>{nb_work_group * nb_work_item},
                                cl::sycl::range<1>{nb_work_item}};
    auto buff  = output_buffer.template get_access<cl::sycl::access::mode::read_write>(cgh);
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
        for(size_t gpos = group_begin + local_id;
            gpos < group_end;
            gpos+=nb_work_item) {
          buff[gpos] = red(acc, buff[gpos]);
        }
      });
    });

  });

  return;
}

template <class ExecutionPolicy, class InputIterator, class OutputIterator,
          class T, class BinaryOperation>
OutputIterator inclusive_scan(ExecutionPolicy &snp, InputIterator b,
                              InputIterator e, OutputIterator o, T init,
                              BinaryOperation bop) {

  cl::sycl::queue q(snp.get_queue());
  auto device = q.get_device();
  size_t size = sycl::helpers::distance(b, e);
  using value_type = typename std::iterator_traits<InputIterator>::value_type;
  {
    cl::sycl::buffer<value_type, 1> buffer { b, e };
    buffer.set_final_data(o);

    auto d = compute_mapscan_descriptor(device, size, sizeof(value_type));
    buffer_mapscan(snp, q, buffer, buffer, init, d, [=](value_type x){return x;}, bop);
  }

  std::advance(o, size);
  return o;
}

#endif
}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_INCLUSIVE_SCAN__
