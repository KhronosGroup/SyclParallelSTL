/* Copyright (c) 2015-2018 The Khronos Group Inc.

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

#ifndef __SYCL_IMPL_ALGORITHM_SORT__
#define __SYCL_IMPL_ALGORITHM_SORT__

#include <type_traits>
#include <typeinfo>
#include <algorithm>

/** sort_kernel_bitonic.
 * Class used to name the bitonic kernel sort per type.
 */
template <typename T>
class sort_kernel_bitonic;

/** sequential_sort_name.
 * Class used to name the bitonic kernel sort per type.
 */
template <typename T>
class sequential_sort_name {
  T userGivenKernelName;
};

/** bitonic_sort_name.
 * Class used to name the bitonic kernel sort per type.
 */
template <typename T>
class bitonic_sort_name {
  T userGivenKernelName;
};

/* sort_swap.
 * Basic simple swap used inside the sort functions.
 */
template <typename T>
void sort_swap(T &lhs, T &rhs) {
  auto temp = rhs;
  rhs = lhs;
  lhs = temp;
}

/* sort_kernel_sequential.
 * Simple kernel to sequentially sort a vector
 */
template <typename T>
class sort_kernel_sequential {
  /* Aliases for SYCL accessors */
  using sycl_rw_acc =
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>;

  sycl_rw_acc a_;
  size_t vS_;

 public:
  sort_kernel_sequential(sycl_rw_acc a, size_t vectorSize)
      : a_(a), vS_(vectorSize){};

  // Simple sequential sort
  void operator()() {
    for (size_t i = 0; i < vS_; i++) {
      for (size_t j = 1; j < vS_; j++) {
        if (a_[j - 1] > a_[j]) {
          sort_swap<T>(a_[j - 1], a_[j]);
        }
      }
    }
  }
};  // class sort_kernel

/* sort_kernel_sequential.
 * Simple kernel to sequentially sort a vector
 */
template <typename T, class ComparableOperator>
class sort_kernel_sequential_comp {
  /* Aliases for SYCL accessors */
  using sycl_rw_acc =
      cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>;

  sycl_rw_acc a_;
  size_t vS_;
  ComparableOperator comp_;

 public:
  sort_kernel_sequential_comp(sycl_rw_acc a, size_t vectorSize,
                              ComparableOperator comp)
      : a_(a), vS_(vectorSize), comp_(comp){};

  // Simple sequential sort
  void operator()() {
    for (size_t i = 0; i < vS_; i++) {
      for (size_t j = 1; j < vS_; j++) {
        if (comp_(a_[j - 1], a_[j])) {
          sort_swap<T>(a_[j - 1], a_[j]);
        }
      }
    }
  }
};  // class sort_kernel

namespace sycl {
namespace impl {

/* Aliases for SYCL accessors */
template <typename T>
using sycl_rw_acc = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                                       cl::sycl::access::target::global_buffer>;

/** isPowerOfTwo.
 * Quick check to ensure num is a power of two.
 * Will only work with integers.
 * @return true if num is power of two
 */
template <typename T>
inline bool isPowerOfTwo(T num) {
  return (num != 0) && !(num & (num - 1));
}

template <>
inline bool isPowerOfTwo<float>(float num) = delete;
template <>
inline bool isPowerOfTwo<double>(double num) = delete;

/** sequential_sort.
 * Command group to call the sequential sort kernel */
template <typename T, typename Alloc>
void sequential_sort(cl::sycl::queue q, cl::sycl::buffer<T, 1, Alloc> buf,
                     size_t vectorSize) {
  auto f = [buf, vectorSize](cl::sycl::handler &h) mutable {
    auto a = buf.template get_access<cl::sycl::access::mode::read_write>(h);
    h.single_task(sort_kernel_sequential<T>(a, vectorSize));
  };
  q.submit(f);
}

/** sequential_sort.
 * Command group to call the sequential sort kernel */
template <typename T, typename Alloc, class ComparableOperator, typename Name>
void sequential_sort(cl::sycl::queue q, cl::sycl::buffer<T, 1, Alloc> buf,
                     size_t vectorSize, ComparableOperator comp) {
  auto f = [buf, vectorSize, comp](cl::sycl::handler &h) mutable {
    auto a = buf.template get_access<cl::sycl::access::mode::read_write>(h);
    h.single_task<Name>(sort_kernel_sequential_comp<T, ComparableOperator>(
        a, vectorSize, comp));
  };
  q.submit(f);
}

#define MULTI_KERNEL_BITONIC_SORT 1
#define EMULATE_SHUFFLE_BUILTINS 1

#ifdef MULTI_KERNEL_BITONIC_SORT
#ifdef EMULATE_SHUFFLE_BUILTINS

namespace emulated_shuffle_builtins {

template <typename vec_type>
typename vec_type::element_type get_vector_component(vec_type &x,
                                                     const unsigned component) {
  using namespace cl::sycl;
  typename vec_type::element_type value;
  switch (component) {
  case 0:
    value = x.x();
    break;
  case 1:
    value = x.y();
    break;
  case 2:
    value = x.z();
    break;
  case 3:
    value = x.w();
    break;
  }
  return value;
}

template <typename vec_type>
void set_vector_component(vec_type &x, const unsigned component,
                          const typename vec_type::element_type value) {
  switch (component) {
  case 0:
    x.x() = value;
    break;
  case 1:
    x.y() = value;
    break;
  case 2:
    x.z() = value;
    break;
  case 3:
    x.w() = value;
    break;
  }
}

static void set_bits32(cl::sycl::cl_uint *const dst, const cl::sycl::cl_uint src,
                       const cl::sycl::cl_uint pos, const cl::sycl::cl_uint len) {
  using namespace cl::sycl;                       
  const cl_uint mask = ((((cl_uint)1) << len) - 1)
                             << pos; // set most significant "len" bits to 1
  const cl_uint shifted_src =
      (src << pos); // shift source value bits to bit offset "pos"
  (*dst) = ((*dst) & ~mask) | (shifted_src & mask);
}

static cl::sycl::cl_uint read_bits32(const cl::sycl::cl_uint src,
                                 const cl::sycl::cl_uint pos,
                                 const cl::sycl::cl_uint len) {
  using namespace cl::sycl;   
  const cl_uint mask = ((1UL << len) - 1) << pos;
  const int dst = (src & mask) >> pos;
  return dst;
}

/**
 * @brief Create a vector containing the components of "x" in the order
 * prescribed by "mask"
 *
 * @param x The vector whose components to reorder.
 * @param mask The vector that determines which input components are placed in
 * the output and the order in which they're placed. The size of components in
 * the mask vector must be the same size as those of the return vector. However,
 * the data type of the mask components must be an unsigned integer type (uchar,
 * ushort, uint, or ulong). NOTE: only a select number of bits in the mask
 * vector's components are important. The k LSBs in each mask component select
 * which input component is placed in the corresponding position in the returned
 * vector. k depends on the number of components in the input vector. Given n
 * components in the input vector, k = {log_{2}}{n}.
 *
 * @return The returned vector will have the same number of components as the
 * mask vector but its element type will be the same as that of the input vector
 * "x".
 *
 */

template <typename gentypem_type, int gentypem_size, typename ugentypen_type,
          int ugentypen_size>
cl::sycl::vec<gentypem_type, gentypem_size>
shuffle(cl::sycl::vec<gentypem_type, gentypem_size> x,
        cl::sycl::vec<ugentypen_type, ugentypen_size> mask) {
  using namespace cl::sycl;
  vec<gentypem_type, gentypem_size> ret;

  const unsigned int k = 2;
  // get k LSBs of components
  vec<ugentypen_type, ugentypen_size> kvec(
      read_bits32(mask.s0(), 0, k), read_bits32(mask.s1(), 0, k),
      read_bits32(mask.s2(), 0, k), read_bits32(mask.s3(), 0, k));

  for (int i = 0; i < 4; ++i) {
    const unsigned int srcIndex = get_vector_component(kvec, i);
    set_vector_component(ret, i, get_vector_component(x, srcIndex));
  }

  return ret;
}

/**
 * @brief
 *
 */
template <typename gentypem_type, int gentypem_size, typename ugentypen_type,
          int ugentypen_size>
cl::sycl::vec<gentypem_type, gentypem_size>
shuffle2(cl::sycl::vec<gentypem_type, gentypem_size> x,
         cl::sycl::vec<gentypem_type, gentypem_size> y,
         cl::sycl::vec<ugentypen_type, ugentypen_size> mask) {
  using namespace cl::sycl;
  vec<gentypem_type, gentypem_size> ret;

  const unsigned int k = 3;
  // get k LSBs of components
  vec<ugentypen_type, ugentypen_size> kvec(
      read_bits32(mask.s0(), 0, k), read_bits32(mask.s1(), 0, k),
      read_bits32(mask.s2(), 0, k), read_bits32(mask.s3(), 0, k));

  for (int i = 0; i < 4; ++i) {
    const unsigned int srcIndex = get_vector_component(kvec, i);
    if (srcIndex < 4) {
      set_vector_component(ret, i, get_vector_component(x, srcIndex));
    } else {
      set_vector_component(ret, i, get_vector_component(y, srcIndex - 4));
    }
  }

  return ret;
}

} // namespace emulated_shuffle_builtins

#endif // #if EMULATE_SHUFFLE_BUILTINS

/* Sort elements within a vector */
#define VECTOR_SORT(input, dir)                                                \
  comp = input < shuffle(input, mask2) ^ dir;                                  \
  input = shuffle(input, (comp * 2 + add2).template as<mask_op_vec_type_>());  \
  comp = input < shuffle(input, mask1) ^ dir;                                  \
  input = shuffle(input, (comp + add1).template as<mask_op_vec_type_>());

#define VECTOR_SWAP(input1, input2, dir)                                       \
  temp = input1;                                                               \
  comp = (((input1 < input2) ^ dir) * 4) + add3;                               \
  input1 = shuffle2(input1, input2, comp.template as<mask_op_vec_type_>());    \
  input2 = shuffle2(input2, temp, comp.template as<mask_op_vec_type_>());

/**
 * @brief
 *
 */
template <typename T, unsigned U = 4> class bitonic_sort_base {
public:
  static_assert(std::is_arithmetic<T>::value,
                "Bitonic sort implementation only works with arithmetic types");
  static_assert(
      U == 4,
      "Bitonic sort implementation only works 4-component vector elements");

  using data_vec_type = cl::sycl::vec<T, U>;
  using relational_op_vec_type = cl::sycl::vec<
      typename std::conditional<
          std::integral_constant<unsigned, sizeof(T)>::value == 1,
          cl::sycl::cl_char,
          typename std::conditional<
              std::integral_constant<unsigned, sizeof(T)>::value == 2,
              cl::sycl::cl_short,
              typename std::conditional<
                  std::integral_constant<unsigned, sizeof(T)>::value == 4,
                  cl::sycl::cl_int, cl::sycl::cl_long>::type>::type>::type,
      U>;

  using mask_op_vec_type = cl::sycl::vec<
      typename std::conditional<
          std::integral_constant<unsigned, sizeof(T)>::value == 1,
          cl::sycl::cl_uchar,
          typename std::conditional<
              std::integral_constant<unsigned, sizeof(T)>::value == 2,
              cl::sycl::cl_ushort,
              typename std::conditional<
                  std::integral_constant<unsigned, sizeof(T)>::value == 4,
                  cl::sycl::cl_uint, cl::sycl::cl_ulong>::type>::type>::type,
      U>;

  typedef std::integral_constant<int, U * 2> data_elems_per_thread_;
  typedef std::integral_constant<int, data_elems_per_thread_::value / U>
      vec_elems_per_thread_;

  using global_buffer_accessor_t =
      cl::sycl::accessor<data_vec_type, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::global_buffer>;
  using local_buffer_accessor_t =
      cl::sycl::accessor<data_vec_type, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>;
};

template <typename T, unsigned U>
class bitonic_sort_init : public bitonic_sort_base<T, U> {
public:
  bitonic_sort_init(
      const typename bitonic_sort_base<T, U>::global_buffer_accessor_t
          &globalBuf,
      const typename bitonic_sort_base<T, U>::local_buffer_accessor_t &localBuf)
      : m_globalBuf(globalBuf), m_localBuf(localBuf) {}

  void operator()(cl::sycl::nd_item<1> item) {
    using namespace cl::sycl;
#ifdef EMULATE_SHUFFLE_BUILTINS
    using namespace emulated_shuffle_builtins;
#endif

    typedef typename bitonic_sort_base<T, U>::data_vec_type data_vec_type_;
    typedef
        typename bitonic_sort_base<T, U>::mask_op_vec_type mask_op_vec_type_;
    typedef typename bitonic_sort_base<T, U>::relational_op_vec_type
        relational_op_vec_type_;

    const int vec_elems_per_thread_ =
        bitonic_sort_base<T, U>::vec_elems_per_thread_::value;

    int dir;
    unsigned int id, global_start, size, stride;

    data_vec_type_ input1, input2, temp;

    relational_op_vec_type_ comp;

    mask_op_vec_type_ mask1(1, 0, 3, 2);
    mask_op_vec_type_ mask2(2, 3, 0, 1);
    mask_op_vec_type_ mask3(3, 2, 1, 0);

    relational_op_vec_type_ add1(1, 1, 3, 3);
    relational_op_vec_type_ add2(2, 3, 2, 3);
    relational_op_vec_type_ add3(1, 2, 2, 3);

    id = item.get_local_id(0) * vec_elems_per_thread_;
    global_start =
        item.get_group(0) * item.get_local_range(0) * vec_elems_per_thread_ +
        id;

    input1 = m_globalBuf[global_start];
    input2 = m_globalBuf[global_start + 1];

    /* Sort input 1 - ascending */
    comp = (input1 < shuffle(input1, mask1));
    input1 = shuffle(input1, (comp + add1).template as<mask_op_vec_type_>());
    comp = (input1 < shuffle(input1, mask2));
    input1 =
        shuffle(input1, (comp * 2 + add2).template as<mask_op_vec_type_>());
    comp = (input1 < shuffle(input1, mask3));
    input1 = shuffle(input1, (comp + add3).template as<mask_op_vec_type_>());

    /* Sort input 2 - descending */
    comp = (input2 > shuffle(input2, mask1));
    input2 = shuffle(input2, (comp + add1).template as<mask_op_vec_type_>());
    comp = (input2 > shuffle(input2, mask2));
    input2 =
        shuffle(input2, (comp * 2 + add2).template as<mask_op_vec_type_>());
    comp = (input2 > shuffle(input2, mask3));
    input2 = shuffle(input2, (comp + add3).template as<mask_op_vec_type_>());

    /* Swap corresponding elements of input 1 and 2 */
    add3 = relational_op_vec_type_(4, 5, 6, 7);
    dir = item.get_local_id(0) % vec_elems_per_thread_ * -1;
    temp = input1;
    comp = (((input1 < input2) ^ dir) * 4 + add3);
    input1 = shuffle2(input1, input2, (comp).template as<mask_op_vec_type_>());
    input2 = shuffle2(input2, temp, (comp).template as<mask_op_vec_type_>());

    /* Sort data and store in local memory */
    VECTOR_SORT(input1, dir);
    VECTOR_SORT(input2, dir);
    m_localBuf[id] = input1;
    m_localBuf[id + 1] = input2;

    /* Create bitonic set */
    for (size = 2; size < item.get_local_range(0); size <<= 1) {
      dir = (item.get_local_id(0) / size & 1) * -1;

      for (stride = size; stride > 1; stride >>= 1) {
        item.barrier(access::fence_space::local_space);
        id = item.get_local_id(0) + (item.get_local_id(0) / stride) * stride;
        VECTOR_SWAP(m_localBuf[id], m_localBuf[id + stride], dir)
      }

      item.barrier(access::fence_space::local_space);
      id = item.get_local_id(0) * vec_elems_per_thread_;
      input1 = m_localBuf[id];
      input2 = m_localBuf[id + 1];
      temp = input1;
      comp = (((input1 < input2) ^ dir) * 4 + add3);
      input1 =
          shuffle2(input1, input2, (comp).template as<mask_op_vec_type_>());
      input2 = shuffle2(input2, temp, (comp).template as<mask_op_vec_type_>());
      VECTOR_SORT(input1, dir);
      VECTOR_SORT(input2, dir);
      m_localBuf[id] = input1;
      m_localBuf[id + 1] = input2;
    }

    /* Perform bitonic merge */
    dir = (item.get_group(0) % vec_elems_per_thread_) * -1;
    for (stride = item.get_local_range(0); stride > 1; stride >>= 1) {
      item.barrier(access::fence_space::local_space);
      id = item.get_local_id(0) + (item.get_local_id(0) / stride) * stride;
      VECTOR_SWAP(m_localBuf[id], m_localBuf[id + stride], dir)
    }
    item.barrier(access::fence_space::local_space);

    /* Perform final sort */
    id = item.get_local_id(0) * vec_elems_per_thread_;
    input1 = m_localBuf[id];
    input2 = m_localBuf[id + 1];

    temp = input1;
    comp = (((input1 < input2) ^ dir) * 4 + add3);
    input1 = shuffle2(input1, input2, (comp).template as<mask_op_vec_type_>());
    input2 = shuffle2(input2, temp, (comp).template as<mask_op_vec_type_>());

    VECTOR_SORT(input1, dir);
    VECTOR_SORT(input2, dir);
    m_globalBuf[global_start] = input1;
    m_globalBuf[global_start + 1] = input2;
  }

private:
  typename bitonic_sort_base<T, U>::global_buffer_accessor_t m_globalBuf;
  typename bitonic_sort_base<T, U>::local_buffer_accessor_t m_localBuf;
};

/**
 * @brief
 *
 */
template <typename T, unsigned U>
class bitonic_sort_stage_0 : public bitonic_sort_base<T, U> {
public:
  bitonic_sort_stage_0(
      const typename bitonic_sort_base<T, U>::global_buffer_accessor_t
          &globalBuf,
      const typename bitonic_sort_base<T, U>::local_buffer_accessor_t &localBuf,
      const unsigned high_stage)
      : m_globalBuf(globalBuf), m_localBuf(localBuf), m_highStage(high_stage) {}

  /**
   * @brief
   *
   */
  void operator()(cl::sycl::nd_item<1> item) {
    using namespace cl::sycl;
#ifdef EMULATE_SHUFFLE_BUILTINS
    using namespace emulated_shuffle_builtins;
#endif

    typedef typename bitonic_sort_base<T, U>::data_vec_type data_vec_type_;
    typedef
        typename bitonic_sort_base<T, U>::mask_op_vec_type mask_op_vec_type_;
    typedef typename bitonic_sort_base<T, U>::relational_op_vec_type
        relational_op_vec_type_;

    const int vec_elems_per_thread_ =
        bitonic_sort_base<T, U>::vec_elems_per_thread_::value;

    int dir;
    unsigned int id, global_start, stride;
    data_vec_type_ input1, input2, temp;
    relational_op_vec_type_ comp;

    mask_op_vec_type_ mask1(1, 0, 3, 2);
    mask_op_vec_type_ mask2(2, 3, 0, 1);
    mask_op_vec_type_ mask3(3, 2, 1, 0);

    relational_op_vec_type_ add1(1, 1, 3, 3);
    relational_op_vec_type_ add2(2, 3, 2, 3);
    relational_op_vec_type_ add3(4, 5, 6, 7);

    /* Determine data location in global memory */
    id = item.get_local_id(0);
    dir = (item.get_group(0) / m_highStage & 1) * -1;
    global_start =
        item.get_group(0) * item.get_local_range(0) * vec_elems_per_thread_ +
        id;

    /* Perform initial swap */
    input1 = m_globalBuf[global_start];
    input2 = m_globalBuf[global_start + item.get_local_range(0)];
    comp = (((input1 < input2) ^ dir) * 4 + add3);
    m_localBuf[id] =
        shuffle2(input1, input2, (comp).template as<mask_op_vec_type_>());
    m_localBuf[id + item.get_local_range(0)] =
        shuffle2(input2, input1, (comp).template as<mask_op_vec_type_>());

    /* Perform bitonic merge */
    for (stride = item.get_local_range(0) / vec_elems_per_thread_; stride > 1;
         stride >>= 1) {
      item.barrier(access::fence_space::local_space);
      id = item.get_local_id(0) + (item.get_local_id(0) / stride) * stride;
      VECTOR_SWAP(m_localBuf[id], m_localBuf[id + stride], dir)
    }

    item.barrier(access::fence_space::local_space);

    /* Perform final sort */
    id = item.get_local_id(0) * vec_elems_per_thread_;
    input1 = m_localBuf[id];
    input2 = m_localBuf[id + 1];
    temp = input1;
    comp = (((input1 < input2) ^ dir) * 4 + add3);
    input1 = shuffle2(input1, input2, (comp).template as<mask_op_vec_type_>());
    input2 = shuffle2(input2, temp, (comp).template as<mask_op_vec_type_>());
    VECTOR_SORT(input1, dir);
    VECTOR_SORT(input2, dir);

    /* Store output in global memory */
    m_globalBuf[global_start + item.get_local_id(0)] = input1;
    m_globalBuf[global_start + item.get_local_id(0) + 1] = input2;
  }

private:
  typename bitonic_sort_base<T, U>::global_buffer_accessor_t m_globalBuf;
  typename bitonic_sort_base<T, U>::local_buffer_accessor_t m_localBuf;
  const int m_highStage;
};

/**
 * @brief
 *
 */
template <typename T, unsigned U>
class bitonic_sort_stage_n : public bitonic_sort_base<T, U> {
public:
  bitonic_sort_stage_n(
      const typename bitonic_sort_base<T, U>::global_buffer_accessor_t
          &globalBuf,
      const typename bitonic_sort_base<T, U>::local_buffer_accessor_t &localBuf,
      const unsigned int stage, const unsigned int high_stage)
      : m_globalBuf(globalBuf), m_localBuf(localBuf), m_stage(stage),
        m_highStage(high_stage) {}

  /**
   * @brief
   *
   */
  void operator()(cl::sycl::nd_item<1> item) {
    using namespace cl::sycl;
#ifdef EMULATE_SHUFFLE_BUILTINS
    using namespace emulated_shuffle_builtins;
#endif

    typedef typename bitonic_sort_base<T, U>::data_vec_type data_vec_type_;
    typedef
        typename bitonic_sort_base<T, U>::mask_op_vec_type mask_op_vec_type_;
    typedef typename bitonic_sort_base<T, U>::relational_op_vec_type
        relational_op_vec_type_;

    int dir;
    data_vec_type_ input1, input2;
    relational_op_vec_type_ comp;
    relational_op_vec_type_ add;
    unsigned int global_start, global_offset;

    add = relational_op_vec_type_(4, 5, 6, 7);

    /* Determine location of data in global memory */
    dir = (item.get_group(0) / m_highStage & 1) * -1;
    global_start =
        (item.get_group(0) + (item.get_group(0) / m_stage) * m_stage) *
            item.get_local_range(0) +
        item.get_local_id(0);
    global_offset = m_stage * item.get_local_range(0);

    /* Perform swap */
    input1 = m_globalBuf[global_start];
    input2 = m_globalBuf[global_start + global_offset];
    comp = (((input1 < input2) ^ dir) * 4 + add);
    m_globalBuf[global_start] =
        shuffle2(input1, input2, (comp).template as<mask_op_vec_type_>());
    m_globalBuf[global_start + global_offset] =
        shuffle2(input2, input1, (comp).template as<mask_op_vec_type_>());
  }

private:
  typename bitonic_sort_base<T, U>::global_buffer_accessor_t m_globalBuf;
  typename bitonic_sort_base<T, U>::local_buffer_accessor_t m_localBuf;
  const unsigned int m_stage;
  const unsigned int m_highStage;
};

/**
 * @brief
 *
 */
template <typename T, unsigned U>
class bitonic_sort_merge : public bitonic_sort_base<T, U> {
public:
  bitonic_sort_merge(
      const typename bitonic_sort_base<T, U>::global_buffer_accessor_t
          &globalBuf,
      const typename bitonic_sort_base<T, U>::local_buffer_accessor_t &localBuf,
      const unsigned int stage, const int dir)
      : m_globalBuf(globalBuf), m_localBuf(localBuf), m_stage(stage),
        mDir(dir) {}

  /**
   * @brief
   *
   */
  void operator()(cl::sycl::nd_item<1> item) {
    using namespace cl::sycl;
#ifdef EMULATE_SHUFFLE_BUILTINS
    using namespace emulated_shuffle_builtins;
#endif

    typedef typename bitonic_sort_base<T, U>::data_vec_type data_vec_type_;
    typedef
        typename bitonic_sort_base<T, U>::mask_op_vec_type mask_op_vec_type_;
    typedef typename bitonic_sort_base<T, U>::relational_op_vec_type
        relational_op_vec_type_;
    const int vec_elems_per_thread_ =
        bitonic_sort_base<T, U>::vec_elems_per_thread_::value;

    data_vec_type_ input1, input2;
    relational_op_vec_type_ comp, add;
    unsigned int global_start, global_offset;

    add = relational_op_vec_type_(4, 5, 6, 7);

    /* Determine location of data in global memory */
    global_start =
        (item.get_group(0) + (item.get_group(0) / m_stage) * m_stage) *
            item.get_local_range(0) +
        item.get_local_id(0);
    global_offset = m_stage * item.get_local_range(0);

    /* Perform swap */
    input1 = m_globalBuf[global_start];
    input2 = m_globalBuf[global_start + global_offset];
    comp = ((input1 < input2 ^ mDir) * 4 + add);
    m_globalBuf[global_start] =
        shuffle2(input1, input2, (comp).template as<mask_op_vec_type_>());
    m_globalBuf[global_start + global_offset] =
        shuffle2(input2, input1, (comp).template as<mask_op_vec_type_>());
  }

private:
  typename bitonic_sort_base<T, U>::global_buffer_accessor_t m_globalBuf;
  typename bitonic_sort_base<T, U>::local_buffer_accessor_t m_localBuf;
  const unsigned int m_stage;
  const int mDir;
};

/**
 * @brief
 *
 */
template <typename T, unsigned U>
class bitonic_sort_merge_last : public bitonic_sort_base<T, U> {
public:
  bitonic_sort_merge_last(
      const typename bitonic_sort_base<T, U>::global_buffer_accessor_t
          &globalBuf,
      const typename bitonic_sort_base<T, U>::local_buffer_accessor_t &localBuf,
      const int dir)
      : m_globalBuf(globalBuf), m_localBuf(localBuf), mDir(dir) {}

  /**
   * @brief
   *
   */
  void operator()(cl::sycl::nd_item<1> item) {
    using namespace cl::sycl;
#ifdef EMULATE_SHUFFLE_BUILTINS
    using namespace emulated_shuffle_builtins;
#endif

    typedef typename bitonic_sort_base<T, U>::data_vec_type data_vec_type_;
    typedef
        typename bitonic_sort_base<T, U>::mask_op_vec_type mask_op_vec_type_;
    typedef typename bitonic_sort_base<T, U>::relational_op_vec_type
        relational_op_vec_type_;
    const int vec_elems_per_thread_ =
        bitonic_sort_base<T, U>::vec_elems_per_thread_::value;

    unsigned int id, global_start, stride;
    data_vec_type_ input1, input2, temp;
    relational_op_vec_type_ comp;

    mask_op_vec_type_ mask1(1, 0, 3, 2);
    mask_op_vec_type_ mask2(2, 3, 0, 1);
    mask_op_vec_type_ mask3(3, 2, 1, 0);

    relational_op_vec_type_ add1(1, 1, 3, 3);
    relational_op_vec_type_ add2(2, 3, 2, 3);
    relational_op_vec_type_ add3(4, 5, 6, 7);

    /* Determine location of data in global memory */
    id = item.get_local_id(0);
    global_start =
        item.get_group(0) * item.get_local_range(0) * vec_elems_per_thread_ +
        id;

    /* Perform initial swap */
    input1 = m_globalBuf[global_start];
    input2 = m_globalBuf[global_start + item.get_local_range(0)];
    comp = ((input1 < input2 ^ mDir) * 4 + add3);
    m_localBuf[id] =
        shuffle2(input1, input2, (comp).template as<mask_op_vec_type_>());
    m_localBuf[id + item.get_local_range(0)] =
        shuffle2(input2, input1, (comp).template as<mask_op_vec_type_>());

    /* Perform bitonic merge */
    for (stride = item.get_local_range(0) / vec_elems_per_thread_; stride > 1;
         stride >>= 1) {
      item.barrier(access::fence_space::local_space);
      id = item.get_local_id(0) + (item.get_local_id(0) / stride) * stride;
      VECTOR_SWAP(m_localBuf[id], m_localBuf[id + stride], mDir)
    }
    item.barrier(access::fence_space::local_space);

    /* Perform final sort */
    id = item.get_local_id(0) * vec_elems_per_thread_;
    input1 = m_localBuf[id];
    input2 = m_localBuf[id + 1];
    temp = input1;
    comp = ((input1 < input2 ^ mDir) * 4 + add3);
    input1 = shuffle2(input1, input2, (comp).template as<mask_op_vec_type_>());
    input2 = shuffle2(input2, temp, (comp).template as<mask_op_vec_type_>());
    VECTOR_SORT(input1, mDir);
    VECTOR_SORT(input2, mDir);

    /* Store the result to global memory */
    m_globalBuf[global_start + item.get_local_id(0)] = input1;
    m_globalBuf[global_start + item.get_local_id(0) + 1] = input2;
  }

private:
  typename bitonic_sort_base<T, U>::global_buffer_accessor_t m_globalBuf;
  typename bitonic_sort_base<T, U>::local_buffer_accessor_t m_localBuf;
  const int mDir;
};

template <typename T, typename Alloc> class kernel_bitonic_sort_init;
template <typename T, typename Alloc> class kernel_bitonic_sort_phase_stage_n;
template <typename T, typename Alloc> class kernel_bitonic_sort_stage_0;
template <typename T, typename Alloc> class kernel_bitonic_sort_merge;
template <typename T, typename Alloc> class kernel_bitonic_sort_merge_last;

/* bitonic_sort.
 * Performs a bitonic sort on the given buffer
 */
template <typename T, typename Alloc>
void bitonic_sort(cl::sycl::queue q, cl::sycl::buffer<T, 1, Alloc> buf,
                  size_t vectorSize) {
  using namespace cl::sycl;
  int direction = 0 /*0 = ascending, -1 = descending*/;

  if (impl::isPowerOfTwo(buf.get_count()) == false) {
    throw std::runtime_error("Buffer size must be a power-of-two");
  }

  if (buf.get_count() < 8) {
    // .. because the bitonic sort allotes 8 elements per work item.
    throw std::runtime_error("Buffer size must be at least 8");
  }

  const std::size_t maxWorkGroupSize =
      q.get_device().get_info<info::device::max_work_group_size>();
  const std::size_t maxLocalMem =
      q.get_device().get_info<info::device::local_mem_size>();
  std::size_t localWorkSize = 2 << static_cast<int>(cl::sycl::log2(
                                  static_cast<float>(maxWorkGroupSize)));

  const std::size_t localMemPerWorkitem =
      sizeof(T) * bitonic_sort_base<T, 4>::data_elems_per_thread_::value;
  std::size_t consumedLocalMem = localWorkSize * localMemPerWorkitem;

  while (consumedLocalMem >= maxLocalMem) {
    localWorkSize /= 2;
    consumedLocalMem = localWorkSize * localMemPerWorkitem;
  }

  const std::size_t globalWorkSize =
      buf.get_count() / bitonic_sort_base<T, 4>::data_elems_per_thread_::value;
  if (globalWorkSize < localWorkSize) {
    localWorkSize = globalWorkSize;
  }

  range<1> reinterpretedRange(buf.get_count() / vec<T, 4>().get_count());
  auto reinterpBuf = buf.template reinterpret<vec<T, 4>, 1>(reinterpretedRange);

  auto ndrange = nd_range<1>(range<1>(globalWorkSize), range<1>(localWorkSize));

  q.submit([&](handler &cgh) {
     auto g = reinterpBuf.template get_access<access::mode::read_write>(cgh);
     typename bitonic_sort_base<T, 4>::local_buffer_accessor_t l(
         range<1>(bitonic_sort_base<T, 4>::vec_elems_per_thread_::value *
                  localWorkSize),
         cgh);

     cgh.parallel_for<kernel_bitonic_sort_init<T, Alloc>>(
         ndrange, bitonic_sort_init<T, 4>(g, l));
   });

  // Execute further stages
  const int num_stages = globalWorkSize / localWorkSize;

  for (cl_uint high_stage = 2; high_stage < num_stages; high_stage <<= 1) {
    for (cl_uint stage = high_stage; stage > 1; stage >>= 1) {
      q.submit([&](handler &cgh) {
         auto g =
             reinterpBuf.template get_access<access::mode::read_write>(cgh);
         typename bitonic_sort_base<T, 4>::local_buffer_accessor_t l(
             range<1>(bitonic_sort_base<T, 4>::vec_elems_per_thread_::value *
                      localWorkSize),
             cgh);

         cgh.parallel_for<kernel_bitonic_sort_phase_stage_n<T, Alloc>>(
             ndrange, bitonic_sort_stage_n<T, 4>(g, l, stage, high_stage));
       })
          .wait();
    }
    q.submit([&](handler &cgh) {
       typename bitonic_sort_base<T, 4>::global_buffer_accessor_t g =
           reinterpBuf.template get_access<access::mode::read_write>(cgh);
       typename bitonic_sort_base<T, 4>::local_buffer_accessor_t l(
           range<1>(bitonic_sort_base<T, 4>::vec_elems_per_thread_::value *
                    localWorkSize),
           cgh);

       cgh.parallel_for<kernel_bitonic_sort_stage_0<T, Alloc>>(
           ndrange, bitonic_sort_stage_0<T, 4>(g, l, high_stage));
     });
  }

  // Perform the bitonic merge
  for (cl_uint stage = num_stages; stage > 1; stage >>= 1) {
    q.submit([&](handler &cgh) {
       typename bitonic_sort_base<T, 4>::global_buffer_accessor_t g =
           reinterpBuf.template get_access<access::mode::read_write>(cgh);
       typename bitonic_sort_base<T, 4>::local_buffer_accessor_t l(
           range<1>(bitonic_sort_base<T, 4>::vec_elems_per_thread_::value *
                    localWorkSize),
           cgh);

       cgh.parallel_for<kernel_bitonic_sort_merge<T, Alloc>>(
           ndrange, bitonic_sort_merge<T, 4>(g, l, stage, direction));
     });
  }

  q.submit([&](handler &cgh) {
     typename bitonic_sort_base<T, 4>::global_buffer_accessor_t g =
         reinterpBuf.template get_access<access::mode::read_write>(cgh);
     typename bitonic_sort_base<T, 4>::local_buffer_accessor_t l(
         range<1>(bitonic_sort_base<T, 4>::vec_elems_per_thread_::value *
                  localWorkSize),
         cgh);

     cgh.parallel_for<kernel_bitonic_sort_merge_last<T, Alloc>>(
         ndrange, bitonic_sort_merge_last<T, 4>(g, l, direction));
   });
}
#else
/* bitonic_sort.
 * Performs a bitonic sort on the given buffer
 */
template <typename T, typename Alloc>
void bitonic_sort(cl::sycl::queue q, cl::sycl::buffer<T, 1, Alloc> buf,
                  size_t vectorSize) {
  int numStages = 0;
  // 2^numStages should be equal to length
  // i.e number of times you halve the lenght to get 1 should be numStages
  for (int tmp = vectorSize; tmp > 1; tmp >>= 1) {
    ++numStages;
  }
  cl::sycl::range<1> r{vectorSize / 2};
  for (int stage = 0; stage < numStages; ++stage) {
    // Every stage has stage + 1 passes
    for (int passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
      auto f = [=](cl::sycl::handler &h) mutable {
        auto a = buf.template get_access<cl::sycl::access::mode::read_write>(h);
        h.parallel_for<sort_kernel_bitonic<T>>(
            cl::sycl::range<1>{r},
            [a, stage, passOfStage](cl::sycl::item<1> it) {
              int sortIncreasing = 1;
              cl::sycl::id<1> id = it.get_id();
              int threadId = id.get(0);

              int pairDistance = 1 << (stage - passOfStage);
              int blockWidth = 2 * pairDistance;

              int leftId = (threadId % pairDistance) +
                           (threadId / pairDistance) * blockWidth;
              int rightId = leftId + pairDistance;

              T leftElement = a[leftId];
              T rightElement = a[rightId];

              int sameDirectionBlockWidth = 1 << stage;

              if ((threadId / sameDirectionBlockWidth) % 2 == 1) {
                sortIncreasing = 1 - sortIncreasing;
              }

              T greater;
              T lesser;

              if (leftElement > rightElement) {
                greater = leftElement;
                lesser = rightElement;
              } else {
                greater = rightElement;
                lesser = leftElement;
              }

              a[leftId] = sortIncreasing ? lesser : greater;
              a[rightId] = sortIncreasing ? greater : lesser;
            });
      };  // command group functor
      q.submit(f);
    }  // passStage
  }    // stage
}  // bitonic_sort
#endif

/* bitonic_sort.
 * Performs a bitonic sort on the given buffer
 */
template <typename T, typename Alloc, class ComparableOperator, typename Name>
void bitonic_sort(cl::sycl::queue q, cl::sycl::buffer<T, 1, Alloc> buf,
                  size_t vectorSize, ComparableOperator comp) {
  int numStages = 0;
  // 2^numStages should be equal to length
  // i.e number of times you halve the lenght to get 1 should be numStages
  for (int tmp = vectorSize; tmp > 1; tmp >>= 1) {
    ++numStages;
  }
  cl::sycl::range<1> r{vectorSize / 2};
  for (int stage = 0; stage < numStages; ++stage) {
    // Every stage has stage + 1 passes
    for (int passOfStage = 0; passOfStage < stage + 1; ++passOfStage) {
      auto f = [=](cl::sycl::handler &h) mutable {
        auto a = buf.template get_access<cl::sycl::access::mode::read_write>(h);
        h.parallel_for<Name>(
            cl::sycl::range<1>{r},
            [a, stage, passOfStage, comp](cl::sycl::item<1> it) {
              int sortIncreasing = 1;
              cl::sycl::id<1> id = it.get_id();
              int threadId = id.get(0);

              int pairDistance = 1 << (stage - passOfStage);
              int blockWidth = 2 * pairDistance;

              int leftId = (threadId % pairDistance) +
                           (threadId / pairDistance) * blockWidth;
              int rightId = leftId + pairDistance;

              T leftElement = a[leftId];
              T rightElement = a[rightId];

              int sameDirectionBlockWidth = 1 << stage;

              if ((threadId / sameDirectionBlockWidth) % 2 == 1) {
                sortIncreasing = 1 - sortIncreasing;
              }

              T greater;
              T lesser;

              if (comp(leftElement, rightElement)) {
                greater = leftElement;
                lesser = rightElement;
              } else {
                greater = rightElement;
                lesser = leftElement;
              }

              a[leftId] = sortIncreasing ? lesser : greater;
              a[rightId] = sortIncreasing ? greater : lesser;
            });
      };  // command group functor
      q.submit(f);
    }  // passStage
  }    // stage
}  // bitonic_sort

template<typename T>
struct buffer_traits;

template<typename T, typename Alloc>
struct buffer_traits<cl::sycl::buffer<T, 1, Alloc>> {
  typedef Alloc allocator_type;
};

/** sort
 * @brief Function that takes a Comp Operator and applies it to the given range
 * @param sep   : Execution Policy
 * @param first : Start of the range
 * @param last  : End of the range
 * @param comp  : Comp Operator
 */
template <class ExecutionPolicy, class RandomIt, class CompareOp>
void sort(ExecutionPolicy &sep, RandomIt first, RandomIt last, CompareOp comp) {
  cl::sycl::queue q(sep.get_queue());
  typedef typename std::iterator_traits<RandomIt>::value_type type_;
  auto buf = std::move(sycl::helpers::make_buffer(first, last));
  auto vectorSize = buf.get_count();

  typedef typename buffer_traits<decltype(buf)>::allocator_type allocator_;
  
  if (impl::isPowerOfTwo(vectorSize)) {
    sycl::impl::bitonic_sort<
        type_, allocator_, CompareOp,
        bitonic_sort_name<typename ExecutionPolicy::kernelName>>(
        q, buf, vectorSize, comp);
  } else {
    sycl::impl::sequential_sort<
        type_, allocator_, CompareOp,
        sequential_sort_name<typename ExecutionPolicy::kernelName>>(
        q, buf, vectorSize, comp);
  }
}

}  // namespace impl
}  // namespace sycl

#endif  // __SYCL_IMPL_ALGORITHM_SORT__
