set(EXAMPLE_FILES
 "parallel_execpol_general.cpp"
 "sycl_sample_00.cpp"
 "sycl_sample_01.cpp" 
 )

foreach(file ${EXAMPLE_FILES})
  build_file(${file})
endforeach()


