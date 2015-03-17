/* Copyright (c) 2015, Codeplay Software Ltd.
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
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
template <typename Iterator>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_buffer_impl(Iterator b, Iterator e, std::random_access_iterator_tag) {
  typedef typename std::iterator_traits<Iterator>::value_type type_;
  size_t bufferSize = std::distance(b, e);
  // We need to copy the data back to the original Iterators when the buffer is
  // destroyed,
  // however there is temporary data midway so we have to somehow force the copy
  // back to the host.
  std::shared_ptr<type_> up{ new type_[bufferSize],
                             [b, bufferSize](type_ *ptr) {
    std::copy(ptr, ptr + bufferSize, b);
    delete[] ptr;
  } };
  std::copy(b, e, up.get());
  cl::sycl::buffer<type_, 1> buf(up, cl::sycl::range<1>(bufferSize));
  buf.set_final_data(up);
  return buf;
}

// Discard buffer
template <typename Iterator>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_buffer_impl(Iterator b, Iterator e, std::input_iterator_tag) {
  typedef typename std::iterator_traits<Iterator>::value_type type_;
  size_t bufferSize = std::distance(b, e);
  std::unique_ptr<type_> up{ new type_[bufferSize] };
  std::copy(b, e, up.get());
  cl::sycl::buffer<type_, 1> buf(std::move(up), cl::sycl::range<1>(bufferSize));
  buf.set_final_data(nullptr);
  return buf;
}

// Iterator range from an existing SYCL buffer
template <typename Iterator>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_buffer_impl(Iterator b, Iterator e, sycl::buffer_iterator_tag) {
  return b.get_buffer();
}

/* Creates a buffer from an iterator range */
template <class Iterator>
cl::sycl::buffer<typename std::iterator_traits<Iterator>::value_type, 1>
make_buffer(Iterator b, Iterator e) {
  return make_buffer_impl(
      b, e, typename std::iterator_traits<Iterator>::iterator_category());
}

} // namespace sycl
} // namespace parallel
} // namespace experimental
} // namespace std

#endif // __EXPERIMENTAL_DETAIL_SYCL_BUFFERS__
