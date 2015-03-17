
add_library(SyclSTL policies.cpp)
add_definitions(-std=c++1y)

if (NOT HEADER_ONLY)
  # From the SYCL package
  build_spir(SyclSTL policies.cpp ${CMAKE_CURRENT_SOURCE_DIR})
  # TODO(Ruyman): build_spir should return the output file name
 set_property(TARGET SyclSTL PROPERTY COMPILE_FLAGS "-std=c++14 -include ${CMAKE_CURRENT_SOURCE_DIR}/policies.cpp.sycl ${CMAKE_CXX_FLAGS}")
endif()

set_property(TARGET SyclSTL PROPERTY CXX_STANDARD "14")
set_property(TARGET SyclSTL PROPERTY CXX_STANDARD_REQUIRED True)

target_link_libraries(SyclSTL ${SYCL_LIBRARY} ${OPENCL_LIBRARY})


