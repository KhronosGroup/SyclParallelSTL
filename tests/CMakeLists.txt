

add_definitions(-std=c++11)

set(GMOCK_DIR "${CMAKE_SOURCE_DIR}/external/gmock/")

if (EXISTS ${GMOCK_DIR}/make/gmock_main.a)
  message(STATUS " GoogleMock found in ${GMOCK_DIR} , building tests ")
  include_directories(${GMOCK_DIR}/include)
  include_directories(${GMOCK_DIR}/gtest/include)

  file(GLOB SRC_FILES "*.cpp")

  if (SRC_FILES)
    add_library(gmock_main STATIC IMPORTED)
    set_property(TARGET gmock_main PROPERTY IMPORTED_LOCATION "${GMOCK_DIR}/make/gmock_main.a")

    foreach(file ${SRC_FILES})
      get_filename_component(filename ${file} NAME_WE)
      message(STATUS "File ${file} Filename ${filename} ")
      add_executable (${filename} ${file})
      target_link_libraries(${filename} gmock_main SyclSTL pthread)

      set_property(TARGET ${filename} PROPERTY CXX_STANDARD "11")
      set_property(TARGET ${filename} PROPERTY CXX_STANDARD_REQUIRED True)

      if (NOT HEADER_ONLY)
        # From the SYCL package
        # TODO(Ruyman): build_spir should return the output file name
        build_spir(${filename} ${file} ${CMAKE_CURRENT_SOURCE_DIR})
        set_property(TARGET ${filename} PROPERTY COMPILE_FLAGS 
                    "-std=c++11 -include ${file}.sycl ${CMAKE_CXX_FLAGS}")
      endif ()

    endforeach()

  else()
    message(WARNING "No tests found")
  endif()


else()
  message(WARNING " Tests are not enabled, GoogleMock not found ")
  message(WARNING " Note that you need to build the gmock_main file in ${GMOCK_DIR}/make")
endif()
