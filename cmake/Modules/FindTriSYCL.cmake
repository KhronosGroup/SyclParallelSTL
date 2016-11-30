#.rst:
# FindTriSYCL
#---------------
#
#   TODO : add Copyright

#########################
#  FindTriSYCL.cmake  
#########################
#
#  Tools for finding and building with triSYCL.
#
#  User must define:
#	TRISYCL_INCLUDE_DIR pointing to triSYCL
#		available : https://github.com/keryell/triSYCL
#	BOOST_COMPUTE_INCLUDE_DIR pointing to Compute
#		available : https://github.com/boostorg/compute
#

# Require CMake version 3.2.2 or higher
cmake_minimum_required(VERSION 3.2.2)

# Check that a supported host compiler can be found
if(CMAKE_COMPILER_IS_GNUCXX)
    # Require at least gcc 4.8
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.8)
      message(FATAL_ERROR
        "host compiler - Not found! (gcc version must be at least 4.8)")
    # Require the GCC dual ABI to be disabled for 5.1 or higher
    elseif (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.1)
      set(COMPUTECPP_DISABLE_GCC_DUAL_ABI "True")
      message(STATUS
        "host compiler - gcc ${CMAKE_CXX_COMPILER_VERSION} (note pre 5.1 gcc ABI enabled)")
    else()
      message(STATUS "host compiler - gcc ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # Require at least clang 3.6
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.6)
      message(FATAL_ERROR
        "host compiler - Not found! (clang version must be at least 3.6)")
    else()
      message(STATUS "host compiler - clang ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
else()
  message(WARNING
	  "host compiler - Not found! (triSYCL supports GCC and Clang, see readme)")
endif()

set(COMPUTECPP_64_BIT_DEFAULT ON)
option(COMPUTECPP_64_BIT_CODE "Compile device code in 64 bit mode"
        ${COMPUTECPP_64_BIT_DEFAULT})
mark_as_advanced(COMPUTECPP_64_BIT_CODE)

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Find triSYCL packagee
if(NOT TRISYCL_INCLUDE_DIR)
	message(FATAL_ERROR "triSYCL package - Not found! (please set TRISYCL_INCLUDE_DIR): ${TRISYCL_INCLUDE_DIR}")
else()
	message(STATUS "triSYCL package - Found")
endif()
option(TRISYCL_INCLUDE_DIR "Path to the triSYCL Package")

# Find Compute packagee
if(NOT BOOST_COMPUTE_INCLUDE_DIR)
	message(FATAL_ERROR "boost/compute package - Not found! (please set BOOST_COMPUTE_INCLUDE_DIR): ${BOOST_COMPUTE_INCLUDE_DIR}")
else()
	message(STATUS "boost/compute package - Found")
endif()
option(BOOST_COMPUTE_INCLUDE_DIR "Path to the boost/compute Package")

# Obtain the triSYCL include directory
set(TRISYCL_INCLUDE_DIRECTORY ${TRISYCL_INCLUDE_DIR})
if (NOT EXISTS ${TRISYCL_INCLUDE_DIRECTORY})
	message(FATAL_ERROR "triSYCL includes - Not found!")
else()
	message(STATUS "triSYCL includes - Found")
endif()

# Obtain the Compute include directory
set(COMPUTE_INCLUDE_DIRECTORY ${BOOST_COMPUTE_INCLUDE_DIR})
if (NOT EXISTS ${COMPUTE_INCLUDE_DIRECTORY})
	message(FATAL_ERROR "boost/compute includes - Not found!")
else()
	message(STATUS "boost/compute includes - Found")
endif()


function(add_sycl_to_target targetName sourceFile binaryDir)
endfunction(add_sycl_to_target)

