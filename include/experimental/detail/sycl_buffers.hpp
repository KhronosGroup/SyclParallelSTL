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
/* vim: set filetype=cpp foldmethod=indent: */
#ifndef __EXPERIMENTAL_DETAIL_SYCL_BUFFERS__
#define __EXPERIMENTAL_DETAIL_SYCL_BUFFERS__

#include <type_traits>
#include <typeinfo>
#include <memory>

#include <experimental/detail/sycl_iterator>

namespace std {
namespace experimental {
namespace parallel {
namespace sycl {

// Buffer with copy-back
template <typename Iterator,
          typename std::enable_if<
              !std::is_base_of<SyclIterator, Iterator>::value>::type* = nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_buffer_impl(Iterator b, Iterator e, std::random_access_iterator_tag) {
  typedef typename std::iterator_traits<Iterator>::value_type type_;
  size_t bufferSize = std::distance(b, e);
  // We need to copy the data back to the original Iterators when the buffer is
  // destroyed,
  // however there is temporary data midway so we have to somehow force the copy
  // back to the host.
  std::shared_ptr<type_> up{new type_[bufferSize], [b, bufferSize](type_* ptr) {
    std::copy(ptr, ptr + bufferSize, b);
    delete[] ptr;
  }};
  std::copy(b, e, up.get());
  cl::sycl::buffer<type_, 1> buf(up, cl::sycl::range<1>(bufferSize));
  buf.set_final_data(up);
  return buf;
}

// Discard buffer
template <typename Iterator,
          typename std::enable_if<
              !std::is_base_of<SyclIterator, Iterator>::value>::type* = nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_buffer_impl(Iterator b, Iterator e, std::input_iterator_tag) {
  typedef typename std::iterator_traits<Iterator>::value_type type_;
  size_t bufferSize = std::distance(b, e);
  std::unique_ptr<type_> up{new type_[bufferSize]};
  std::copy(b, e, up.get());
  cl::sycl::buffer<type_, 1> buf(std::move(up), cl::sycl::range<1>(bufferSize));
  buf.set_final_data(nullptr);
  return buf;
}

// Iterator range from an existing SYCL buffer
template <typename Iterator>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
reuse_buffer_impl(Iterator b, Iterator e, std::input_iterator_tag) {
  // TODO: This may need to create a sub-buffer if the range does not match
  //  the whole buffer.
  //  TODO: Technically this can be a const buffer since it is input-only
  return b.get_buffer();
}

// Iterator range from an existing SYCL buffer
template <typename Iterator>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
reuse_buffer_impl(Iterator b, Iterator e, std::random_access_iterator_tag) {
  // TODO: This may need to create a sub-buffer if the range does not match
  //  the whole buffer.
  return b.get_buffer();
}

/* Creates a buffer from an iterator range */
template <class Iterator, typename std::enable_if<std::is_base_of<
                              SyclIterator, Iterator>::value>::type* = nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_buffer(Iterator b, Iterator e) {
  return reuse_buffer_impl(
      b, e, typename std::iterator_traits<Iterator>::iterator_category());
}

template <class Iterator,
          typename std::enable_if<
              !std::is_base_of<SyclIterator, Iterator>::value>::type* = nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_buffer(Iterator b, Iterator e) {
  return make_buffer_impl(
      b, e, typename std::iterator_traits<Iterator>::iterator_category());
}

/* Creates a read only constant buffer from an iterator range */
template <class Iterator, typename std::enable_if<std::is_base_of<
                              SyclIterator, Iterator>::value>::type* = nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_const_buffer(Iterator b, Iterator e) {
  return reuse_buffer_impl(b, e, std::input_iterator_tag());
}

template <class Iterator,
          typename std::enable_if<
              !std::is_base_of<SyclIterator, Iterator>::value>::type* = nullptr>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_const_buffer(Iterator b, Iterator e) {
  return make_buffer_impl(b, e, std::input_iterator_tag());
}

}  // namespace sycl
}  // namespace parallel
}  // namespace experimental
}  // namespace std

#endif  // __EXPERIMENTAL_DETAIL_SYCL_BUFFERS__
