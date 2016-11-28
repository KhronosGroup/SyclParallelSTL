cmake_minimum_required (VERSION 3.2.2)
project (SyclSTL)

enable_testing()

option(PARALLEL_STL_BENCHMARKS "Build the internal benchmarks" OFF)

message(STATUS " Path to CMAKE source directory: ${CMAKE_SOURCE_DIR} ")
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules/)

if (USE_COMPUTECPP)
	message(STATUS " Using ComputeCpp CMake")

	message(STATUS " Path to ComputeCpp implementation: ${COMPUTECPP_PACKAGE_ROOT_DIR} ")

	include(FindOpenCL)
	include(FindComputeCpp)

	include_directories("${COMPUTECPP_INCLUDE_DIRECTORY}")

else()
	message(STATUS " Using triSYCL CMake")

	add_compile_options("-std=c++14") 

	message(STATUS " Path to TriSycl implementation: ${TRISYCL_PACKAGE_ROOT_DIR} ")
	message(STATUS " Path to boost/compute implementation: ${COMPUTE_PACKAGE_ROOT_DIR} ")

	include(FindOpenCL)
	include(FindTriSycl)
	
	include_directories("${COMPUTE_INCLUDE_DIRECTORY}")
	include_directories("${TRISYCL_INCLUDE_DIRECTORY}")

endif()


# PSTL specific 
include_directories("include")

add_subdirectory (src)
add_subdirectory (examples)
add_subdirectory (tests)

if (PARALLEL_STL_BENCHMARKS)
  add_subdirectory (benchmarks)
endif()
