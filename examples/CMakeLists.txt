

file(GLOB EXAMPLE_FILES ./*.cpp)

foreach(file ${EXAMPLE_FILES})
  get_filename_component(filename ${file} NAME_WE)
  message(STATUS "File ${file} Filename ${filename} ")
  add_executable (${filename} ${file})
  target_link_libraries(${filename} SyclSTL)

  set_property(TARGET ${filename} PROPERTY CXX_STANDARD "1y")
  set_property(TARGET ${filename} PROPERTY CXX_STANDARD_REQUIRED True)

  if (NOT HEADER_ONLY)
    # From the SYCL package
    # TODO(Ruyman): build_spir should return the output file name
    build_spir(${filename} ${file} ${CMAKE_CURRENT_SOURCE_DIR})
    set_property(TARGET ${filename} PROPERTY COMPILE_FLAGS 
                "-std=c++1y -include ${file}.sycl ${CMAKE_CXX_FLAGS}")
  endif ()

endforeach()
