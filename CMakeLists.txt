cmake_minimum_required (VERSION 3.2.2)
project (SyclSTL)

enable_testing()

option(PARALLEL_STL_BENCHMARKS "Build the internal benchmarks" OFF)
option(USE_COMPUTECPP "Use ComputeCPP" ON)

message(STATUS " Path to CMAKE source directory: ${CMAKE_SOURCE_DIR} ")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)

if (USE_COMPUTECPP)
  message(STATUS " Using ComputeCpp CMake")

  include(FindOpenCL)
  include(FindComputeCpp)

  add_definitions(-DSYCL_PSTL_USE_OLD_ALGO)
  set(SYCL_LIBRARY ${COMPUTECPP_RUNTIME_LIBRARY})
  set(SYCL_INCLUDE_DIRS ${ComputeCpp_INCLUDE_DIRS})
else()
  message(STATUS " Using triSYCL CMake")
  include(FindTriSYCL)
  set(SYCL_INCLUDE_DIRS ${TRISYCL_INCLUDE_DIR})
  set(SYCL_LIBRARY)  # Not needed for TriSYCL
endif()


# PSTL specific
include_directories("include")

# Function to build a file 
function(build_file file)
  get_filename_component(SOURCE_NAME ${file} NAME_WE)
  message(STATUS " Adding ${SOURCE_NAME} ")

  add_executable(${SOURCE_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${file} )

  add_sycl_to_target(TARGET ${SOURCE_NAME} SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endfunction()

add_subdirectory (src)
add_subdirectory (examples)
add_subdirectory (tests)

if (PARALLEL_STL_BENCHMARKS)
  add_subdirectory (benchmarks)
endif()
