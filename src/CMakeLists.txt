
add_library(SyclSTL policies.cpp)

set_property(TARGET SyclSTL PROPERTY CXX_STANDARD "14")
set_property(TARGET SyclSTL PROPERTY CXX_STANDARD_REQUIRED True)

set_property(TARGET SyclSTL APPEND PROPERTY COMPILE_FLAGS "-std=c++14")

target_link_libraries(SyclSTL ${SYCL_LIBRARY}  ${OPENCL_LIBRARY})

