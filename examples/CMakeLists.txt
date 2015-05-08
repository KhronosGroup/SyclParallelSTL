

file(GLOB EXAMPLE_FILES ./*.cpp)

foreach(file ${EXAMPLE_FILES})
  get_filename_component(filename ${file} NAME)
  set(exe_name "${filename}.exe")
  message(STATUS "File ${file} Filename ${filename}  Exename ${exe_name}")
  add_executable (${exe_name} ${file})
  target_link_libraries(${exe_name} SyclSTL)

  set_property(TARGET ${exe_name} PROPERTY CXX_STANDARD "1y")
  set_property(TARGET ${exe_name} PROPERTY CXX_STANDARD_REQUIRED True)

  if (NOT HEADER_ONLY)
    # From the SYCL package
    # TODO(Ruyman): build_spir should return the output file name
    build_spir(${exe_name} ${file} ${CMAKE_CURRENT_BINARY_DIR})
    if (MSVC)
     set_property(TARGET ${exe_name} PROPERTY COMPILE_FLAGS 
       "-std=c++1y /FI ${CMAKE_CURRENT_BINARY_DIR}/${filename}.sycl ${CMAKE_CXX_FLAGS}")
     else(MSVC)
       set_property(TARGET ${exe_name} PROPERTY COMPILE_FLAGS 
      "-std=c++1y -include ${CMAKE_CURRENT_BINARY_DIR}/${filename}.sycl ${CMAKE_CXX_FLAGS}")
    endif(MSVC)
    add_dependencies(${exe_name} ${file}.sycl)
 
  endif ()

endforeach()
