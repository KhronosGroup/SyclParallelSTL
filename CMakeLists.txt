cmake_minimum_required (VERSION 3.2.2)
project (SyclSTL)

enable_testing()

option(PARALLEL_STL_BENCHMARKS "Build the internal benchmarks" OFF)

message(STATUS " Path to SYCL implementation: ${COMPUTECPP_PACKAGE_ROOT_DIR} ")

if(SYCL_NO_DEVICE_COMPILER)
  message(NOTE "The SYCL implementation has no device compiler")
  add_definitions(-DSYCL_NO_DEVICE_COMPILER=1)
else(SYCL_NO_DEVICE_COMPILER)
  #####  Find the SYCL implementation

  find_library(SYCL_LIBRARY
    NAMES SYCL SYCL_d
    HINTS ${COMPUTECPP_PACKAGE_ROOT_DIR}
    PATH_SUFFIXES lib
    )

  find_library(OPENCL_LIBRARY
    NAMES OpenCL
    HINTS ${OPENCL_ROOT_DIR}
    PATH_SUFFIXES lib lib64
    )

  if(NOT DEFINED OPENCL_LIBRARY)
    message(FATAL_ERROR "OpenCL library is not defined")
  endif()

  message(STATUS "${SYCL_LIBRARY}")
  if (${SYCL_LIBRARY} MATCHES "NOTFOUND")
    message(FATAL_ERROR "Cannot find SYCL implementation (${COMPUTECPP_PACKAGE_ROOT_DIR})")
  else()
    message(STATUS "The path to SYCL is: ${COMPUTECPP_PACKAGE_ROOT_DIR}")
  endif()

  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")


  set(DEVICE_COMPILER_PATH "${COMPUTECPP_PACKAGE_ROOT_DIR}/bin/")
  set(DEVICE_COMPILER_INCLUDE_PATH "${COMPUTECPP_PACKAGE_ROOT_DIR}/include/device_compiler")
  set(RUNTIME_INCLUDES "${COMPUTECPP_PACKAGE_ROOT_DIR}/include")

  if (UNIX)
      # TODO(Ruyman) Figure out the linux distribution
      # Fix for device compiler not finding specific C++ headers with Ubuntu
      set(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH}"
                         "/usr/include/c++/4.8.2/"
                          "/usr/include/x86_64-linux-gnu/c++/4.8/")
  endif()

  if (NOT DEFINED DEVICE_COMPILER_NAME)
    if (WIN32)
      set(DEVICE_COMPILER_NAME "compute++.exe")
    else(WIN32)
      set(DEVICE_COMPILER_NAME "compute++")
    endif(WIN32)
  endif (NOT DEFINED DEVICE_COMPILER_NAME)
  set(DEVICE_COMPILER "${DEVICE_COMPILER_PATH}/${DEVICE_COMPILER_NAME}"
    CACHE FILEPATH "device compiler")
  message(STATUS "### Device Compiler ${DEVICE_COMPILER}")
  add_custom_target(compiler)

  # set(COMPUTECPP_PACKAGE_ROOT_DIR ${COMPUTECPP_PACKAGE_ROOT_DIR})
  set(OpenCL_INCLUDE_DIR ${COMPUTECPP_PACKAGE_ROOT_DIR}/include/CL/)
  include(FindComputeCpp)
  set(HOST_COMPILER_OPTIONS -O3 -std=c++11)

  add_library(SYCL_LIBRARY SHARED IMPORTED)
  add_library(OPENCL_LIBRARY SHARED IMPORTED)

endif(SYCL_NO_DEVICE_COMPILER)

include_directories("${COMPUTECPP_PACKAGE_ROOT_DIR}/include")
include_directories("${COMPUTECPP_PACKAGE_ROOT_DIR}/include/image_library")

# Build the parallel stl

include_directories("include")

add_subdirectory (src)
add_subdirectory (examples)

add_subdirectory (tests)

if (PARALLEL_STL_BENCHMARKS)
  add_subdirectory (benchmarks)
endif()
