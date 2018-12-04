set(BENCHMARK_FILES 
  "basic.cpp"
  "montecarloPI.cpp"
  "nbody.cpp"
  "std_foreach_saxpy.cpp"
  "std_sort.cpp"
  "std_transform_saxpy.cpp"
  "sycl_exclusive_scan.cpp"
  "sycl_foreach_saxpy.cpp"
  "sycl_inclusive_scan.cpp"
  "sycl_reduce.cpp"
  "sycl_sort.cpp"
  "sycl_transform_het_saxpy.cpp"
  "sycl_transform_reduce_saxpy.cpp"
  "sycl_transform_saxpy.cpp"
  "word_count.cpp"
)


foreach(file ${BENCHMARK_FILES})
  build_file(${file})
endforeach()
