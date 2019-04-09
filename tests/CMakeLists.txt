include(ExternalProject)

set(SYCLBLAS_TEST ${CMAKE_CURRENT_SOURCE_DIR})
set(SYCLBLAS_TEST_INCLUDE "${SYCLBLAS_TEST}/include")

if(CMAKE_CROSSCOMPILING)
    set(cmake_toolchain
      "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
       )
     message(STATUS "Crossing : ${cmake_toolchain}")
endif()

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)

if(VERBOSE)
  add_definitions(-DVERBOSE=VERBOSE)
endif(VERBOSE)

if(SYCL_DEVICE)
  add_definitions(-DSYCL_DEVICE=${SYCL_DEVICE})
endif(SYCL_DEVICE)

include_directories(
  ${GTEST_LIB}
  ${SYCL_INCLUDE_DIRS}
)


add_subdirectory(pstl-tests)
